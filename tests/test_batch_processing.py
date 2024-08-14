import pytest
from pathlib import Path
import json
import os
from src.batch_processing import upload_batch_file, create_batch_job, wait_for_batch_completion, process_batch_results
from src.batch_preparation import prepare_batch_file
from pydantic import BaseModel
import time
from tests.testing_prompts import PROMPTS, SYSTEM_PROMPT

class ExampleResponseModel(BaseModel):
    content: str

@pytest.fixture(scope="module")
def batch_files():
    # Define paths
    input_file = Path("data/raw/batch_inputs/example_input.jsonl")
    output_file = Path("data/raw/batch_outputs/example_output.jsonl")
    
    # Prepare a batch file
    prompts = PROMPTS
    system_message = SYSTEM_PROMPT
    batch_file = prepare_batch_file(
        prompts=prompts,
        response_model=ExampleResponseModel,
        system_message=system_message,
        model="gpt-4o-2024-08-06",
        max_tokens=2000,
        save_dir=str(input_file.parent)
    )
    
    # Rename the batch file to example_input.jsonl
    os.rename(batch_file, input_file)

    # Upload the batch file
    file_id = upload_batch_file(input_file)
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
    processed_output_file = process_batch_results(completed_batch.id, output_file.parent)
    
    # Rename the output file to example_output.jsonl
    os.rename(processed_output_file, output_file)
    
    return input_file, output_file

def test_full_batch_processing_workflow(batch_files):
    input_file, output_file = batch_files
    assert input_file.exists()
    assert output_file.exists()
    assert input_file.stat().st_size > 0
    assert output_file.stat().st_size > 0

    print(f"\nExamining input file: {input_file}")
    with open(input_file, 'r') as f:
        input_content = f.read()
        print(input_content)

    print(f"\nExamining output file: {output_file}")
    with open(output_file, 'r') as f:
        output_content = f.read()
        print(output_content)

    print("Full batch processing workflow completed successfully.")

def test_validate_batch_output(batch_files):
    _, output_file = batch_files
    with open(output_file, 'r') as f:
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
            
            content = result['response']['body']['choices'][0]['message']['content']
            ExampleResponseModel(content=content)
    
    print("Batch output validation completed successfully.")