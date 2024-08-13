import unittest
from pathlib import Path
import json
from pydantic import BaseModel
from src.batch_preparation import format_batch_request, save_to_jsonl, prepare_batch_file

class TestResponseModel(BaseModel):
    content: str

class TestBatchPreparation(unittest.TestCase):
    def setUp(self):
        self.prompts = ["What's the capital of France?", "Who wrote 'Romeo and Juliet'?"]
        self.system_message = "You are a helpful assistant."
        self.model = "gpt-4"
        self.max_tokens = 1000
        self.test_dir = Path(__file__).parent / "test_data"
        self.test_dir.mkdir(exist_ok=True)

    def test_format_batch_request(self):
        result = format_batch_request(
            self.prompts,
            self.system_message,
            TestResponseModel,
            self.model,
            self.max_tokens
        )
        
        self.assertEqual(len(result), len(self.prompts))
        for i, item in enumerate(result):
            self.assertEqual(item['custom_id'], f"request-{i+1}")
            self.assertEqual(item['method'], "POST")
            self.assertEqual(item['url'], "/v1/chat/completions")
            self.assertEqual(item['body']['model'], self.model)
            self.assertEqual(item['body']['messages'][0]['role'], "system")
            self.assertEqual(item['body']['messages'][0]['content'], self.system_message)
            self.assertEqual(item['body']['messages'][1]['role'], "user")
            self.assertEqual(item['body']['messages'][1]['content'], self.prompts[i])
            self.assertEqual(item['body']['max_tokens'], self.max_tokens)
            self.assertIn('response_format', item['body'])
            self.assertEqual(item['body']['response_format']['name'], "TestResponseModel")
            self.assertFalse(item['body']['response_format']['schema']['additionalProperties'])

    def test_save_to_jsonl(self):
        data = [{"key": "value1"}, {"key": "value2"}]
        test_file = self.test_dir / "test_save.jsonl"
        save_to_jsonl(data, str(test_file))

        self.assertTrue(test_file.exists())
        with open(test_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(data))
            for i, line in enumerate(lines):
                self.assertEqual(json.loads(line), data[i])

    def test_prepare_batch_file(self):
        result_path = prepare_batch_file(
            self.prompts,
            TestResponseModel,
            self.system_message,
            self.model,
            self.max_tokens,
            str(self.test_dir)
        )

        self.assertTrue(result_path.exists())
        self.assertEqual(result_path.name, "batch_input.jsonl")
        
        with open(result_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(self.prompts))
            for line in lines:
                data = json.loads(line)
                self.assertIn('custom_id', data)
                self.assertIn('method', data)
                self.assertIn('url', data)
                self.assertIn('body', data)

    def tearDown(self):
        # Clean up test files
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()

if __name__ == '__main__':
    unittest.main()