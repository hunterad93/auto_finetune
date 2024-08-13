from typing import Dict, Any, List
from pathlib import Path
import time
import json
from src.utils import get_openai_client

def upload_batch_file(file_path: Path) -> str:
    """Upload the batch input file and return the file ID."""
    client = get_openai_client()
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    return file.id

def create_batch_job(file_id: str) -> str:
    """Create a batch job and return the batch ID."""
    client = get_openai_client()
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Batch job created with ID: {batch.id}")
    return batch.id

def wait_for_batch_completion(batch_id: str, polling_interval: int = 60) -> Dict[str, Any]:
    """Poll for batch job completion and return the completed batch object."""
    client = get_openai_client()
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return batch
        elif batch.status in ["failed", "expired", "cancelled"]:
            raise Exception(f"Batch job {batch_id} {batch.status}")
        print(f"Batch status: {batch.status}. Waiting {polling_interval} seconds...")
        time.sleep(polling_interval)

def process_batch_results(batch: Dict[str, Any], save_dir: str) -> List[Dict[str, Any]]:
    """Retrieve and process batch results, saving them to a JSONL file."""
    client = get_openai_client()
    output_file_content = client.files.content(batch.output_file_id)
    
    processed_data = []
    for line in output_file_content.splitlines():
        result = json.loads(line)
        if result['response']['status_code'] == 200:
            assistant_content = result['response']['body']['choices'][0]['message']['content']
            data_point = {
                "messages": [
                    {"role": "system", "content": result['response']['body']['messages'][0]['content']},
                    {"role": "user", "content": result['response']['body']['messages'][1]['content']},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            processed_data.append(data_point)
        else:
            print(f"Error in batch response for custom_id {result['custom_id']}: {result['response']['status_code']}")

    project_root = Path(__file__).parent.parent
    full_save_dir = project_root / save_dir
    output_file_path = full_save_dir / "processed_responses.jsonl"
    
    with open(output_file_path, 'w') as f:
        for item in processed_data:
            json.dump(item, f)
            f.write('\n')
    
    print(f"Processed and saved {len(processed_data)} responses to {output_file_path}")

    return processed_data

# Example usage
if __name__ == "__main__":
    from src.batch_preparation import prepare_batch_file
    from pydantic import BaseModel

    class ExampleResponseModel(BaseModel):
        content: str

    prompts = ["What's the capital of France?", "Who wrote 'Romeo and Juliet'?"]
    system_message = "You are a helpful assistant."
    save_dir = "data/raw"

    # Prepare batch file
    batch_file_path = prepare_batch_file(
        prompts=prompts,
        response_model=ExampleResponseModel,
        system_message=system_message,
        save_dir=save_dir
    )

    # Upload batch file and create job
    file_id = upload_batch_file(batch_file_path)
    batch_id = create_batch_job(file_id)

    # Wait for completion and process results
    completed_batch = wait_for_batch_completion(batch_id)
    processed_data = process_batch_results(completed_batch, save_dir)

    print(f"Collected and processed {len(processed_data)} data points using batch API.")