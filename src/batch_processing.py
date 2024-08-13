import shutil
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

def process_batch_results(batch_id: str, save_dir: Path) -> Path:
    """
    Retrieve the batch results file and save it to the specified directory.
    Returns the path to the saved file.
    """
    client = get_openai_client()
    
    # Retrieve the batch to get the output file ID
    batch_info = client.batches.retrieve(batch_id)
    output_file_id = batch_info.output_file_id
    
    # Ensure the save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path
    output_file_path = save_dir / f"batch_output_{batch_id}.jsonl"
    
    # Download and save the file
    response = client.files.content(output_file_id)
    with open(output_file_path, 'wb') as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    
    print(f"Batch results saved to: {output_file_path}")
    
    return output_file_path