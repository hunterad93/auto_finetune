import pytest
from pathlib import Path
import json
import os
from src.batch_processing import upload_batch_file, create_batch_job, wait_for_batch_completion, process_batch_results
from src.batch_preparation import prepare_batch_file
from pydantic import BaseModel
import time

class ExampleResponseModel(BaseModel):
    content: str

@pytest.fixture(scope="module")
def batch_output_file(request):
    # Prepare a batch file
    prompts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ]
    system_message = "You are a helpful assistant."
    batch_file = prepare_batch_file(
        prompts=prompts,
        response_model=ExampleResponseModel,
        system_message=system_message,
        model="gpt-4o-2024-08-06",
        max_tokens=2000,
        save_dir="data/raw/batch_inputs"
    )

    # Upload the batch file
    file_id = upload_batch_file(batch_file)
    assert file_id is not None
    print(f"Uploaded batch file. File ID: {file_id}")

    # Create a batch job
    batch_id = create_batch_job(file_id)
    assert batch_id is not None
    print(f"Created batch job. Batch ID: {batch_id}")

    # Wait for batch completion
    completed_batch = wait_for_batch_completion(batch_id, polling_interval=10)
    assert completed_batch.status == "completed"
    print("Batch job completed successfully.")

    # Process batch results
    output_dir = Path("data/raw/batch_outputs")
    output_file = process_batch_results(completed_batch.id, output_dir)
    
    def finalizer():
        print(f"Cleaning up test files...")
        if os.path.exists(batch_file):
            os.remove(batch_file)
        if os.path.exists(output_file):
            os.remove(output_file)
    
    request.addfinalizer(finalizer)
    
    return output_file

def test_full_batch_processing_workflow(batch_output_file):
    assert batch_output_file.exists()
    assert batch_output_file.stat().st_size > 0  # Check that the file is not empty

    # Examine the output file
    print(f"\nExamining output file: {batch_output_file}")
    with open(batch_output_file, 'r') as f:
        output_content = f.read()
        print(output_content)

    print("Full batch processing workflow completed successfully.")

def test_validate_batch_output(batch_output_file):
    # Read and validate the output
    with open(batch_output_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            assert 'custom_id' in result
            assert 'response' in result
            assert 'status_code' in result['response']
            assert result['response']['status_code'] == 200
            assert 'body' in result['response']
            assert 'choices' in result['response']['body']
            assert len(result['response']['body']['choices']) > 0
            assert 'message' in result['response']['body']['choices'][0]
            assert 'content' in result['response']['body']['choices'][0]['message']
            
            # Validate content based on TestResponseModel
            content = result['response']['body']['choices'][0]['message']['content']
            ExampleResponseModel(content=content)  # This will raise a validation error if content is invalid
    
    print("Batch output validation completed successfully.")