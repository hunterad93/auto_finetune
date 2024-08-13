import pytest
from pathlib import Path
import json
from src.data_processing import prepare_finetuning_data, validate_finetuning_data

@pytest.fixture
def example_paths():
    return {
        'input': Path('data/raw/batch_inputs/example_input.jsonl'),
        'output': Path('data/raw/batch_outputs/example_output.jsonl'),
        'processed': Path('data/processed/finetuningJSONLs')
    }

def test_prepare_finetuning_data(example_paths):
    # Define the output file path
    output_file = example_paths['processed'] / "example_finetune.jsonl"
    
    try:
        # Prepare the fine-tuning data
        result_file = prepare_finetuning_data(
            example_paths['input'],
            example_paths['output'],
            output_dir=example_paths['processed'],
            output_filename="example_finetune.jsonl"
        )
    except Exception as e:
        pytest.fail(f"prepare_finetuning_data raised an exception: {e}")
    
    # Check if the file was created
    assert result_file.exists(), f"Result file was not created at {result_file}"
    assert result_file == output_file, f"Result file path {result_file} does not match expected path {output_file}"
    
    # Validate the structure of the created file
    assert validate_finetuning_data(result_file), f"Created file at {result_file} is not valid"
    
    # Check the content of the created file
    with open(result_file, 'r') as f:
        finetuning_data = [json.loads(line) for line in f]
    
    # Assert we have the correct number of items
    assert len(finetuning_data) == 2, f"Expected 2 items, but got {len(finetuning_data)}"
    
    # Check the structure of each item
    for i, item in enumerate(finetuning_data):
        assert isinstance(item, dict), f"Item {i} is not a dictionary"
        assert list(item.keys()) == ['messages'], f"Item {i} should only have a 'messages' key, but has keys: {list(item.keys())}"
        assert isinstance(item['messages'], list), f"Item {i}'s 'messages' value is not a list"
        assert len(item['messages']) == 3, f"Item {i} should have 3 messages, but has {len(item['messages'])}"
        
        roles = [msg['role'] for msg in item['messages']]
        assert roles == ['system', 'user', 'assistant'], f"Item {i} has incorrect message roles: {roles}"
        
        for j, msg in enumerate(item['messages']):
            assert set(msg.keys()) == {'role', 'content'}, f"Message {j} in item {i} has incorrect keys: {set(msg.keys())}"
            assert isinstance(msg['role'], str), f"Role in message {j} of item {i} is not a string"
            assert isinstance(msg['content'], str), f"Content in message {j} of item {i} is not a string"
            assert msg['content'].strip(), f"Content in message {j} of item {i} is empty or only whitespace"
    
    print(f"Fine-tuning data successfully created and validated at: {result_file}")