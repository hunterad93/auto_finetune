from typing import List, Type, Dict, Any
from pydantic import BaseModel
from src.utils import pydantic_to_json_schema, save_to_jsonl
from pathlib import Path

def format_batch_request(
    prompts: List[str],
    system_message: str,
    response_model: Type[BaseModel],
    model: str,
    max_tokens: int
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
                "response_format": response_format
            }
        }
        for i, prompt in enumerate(prompts, start=1)
    ]

def prepare_batch_file(
    prompts: List[str],
    response_model: Type[BaseModel],
    system_message: str,
    model: str,
    max_tokens: int,
    save_dir: Path,
    filename_prefix: str
) -> Path:
    """Prepare batch request file and return its path."""
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_requests = format_batch_request(prompts, system_message, response_model, model, max_tokens)
    input_file_path = save_dir / f"{filename_prefix}_batch_input.jsonl"
    save_to_jsonl(batch_requests, str(input_file_path))

    print(f"Batch input file created at: {input_file_path}")
    return input_file_path