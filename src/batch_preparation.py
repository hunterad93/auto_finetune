from typing import List, Type, Dict, Any
from pydantic import BaseModel
from src.utils import pydantic_to_json_schema, save_to_jsonl
from pathlib import Path

def format_batch_request(
    prompts: List[str],
    system_message: str,
    response_model: Type[BaseModel],
    model: str,
    max_tokens: int = 2000
) -> List[Dict[str, Any]]:
    """
    Format requests for the batch API input file.
    """
    response_format = pydantic_to_json_schema(response_model)
    return [
        {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "response_format": response_format["json_schema"]
            }
        }
        for i, prompt in enumerate(prompts, start=1)
    ]

def prepare_batch_file(
    prompts: List[str],
    response_model: Type[BaseModel],
    system_message: str,
    model: str = "gpt-4o-2024-08-06",
    max_tokens: int = 2000,
    save_dir: str = "data/raw/batch_inputs"
) -> Path:
    """Prepare batch request file and return its path."""
    project_root = Path(__file__).parent.parent
    full_save_dir = project_root / save_dir
    full_save_dir.mkdir(parents=True, exist_ok=True)

    batch_requests = format_batch_request(prompts, system_message, response_model, model, max_tokens)
    input_file_path = full_save_dir / "batch_input.jsonl"
    save_to_jsonl(batch_requests, str(input_file_path))

    print(f"Batch input file created at: {input_file_path}")
    return input_file_path

# Example usage
if __name__ == "__main__":
    from pydantic import BaseModel

    class ExampleResponseModel(BaseModel):
        content: str

    prompts = ["What's the capital of France?", "Who wrote 'Romeo and Juliet'?"]
    system_message = "You are a helpful assistant."

    prepare_batch_file(
        prompts=prompts,
        response_model=ExampleResponseModel,
        system_message=system_message,
        model="gpt-4o-2024-08-06"
    )