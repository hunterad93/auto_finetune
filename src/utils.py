from typing import Type, Any, Dict, List
from pydantic import BaseModel
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv('/Users/adamhunter/miniconda3/envs/ragdev/ragdev.env')

def get_openai_client() -> OpenAI:
    """Create and return an OpenAI client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to a JSON schema for OpenAI's structured API."""
    schema = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
                "additionalProperties": False
            },
            "strict": True
        }
    }

def save_to_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data to a JSONL file."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def call_openai_api(
    messages: list,
    model: str,
    response_model: Type[BaseModel],
    temperature: float = 0.0,
    timeout: int = 60
) -> Any:
    """Make a call to the OpenAI API using structured output."""
    client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=pydantic_to_json_schema(response_model),
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        raise