from pathlib import Path
from typing import Optional
from src.utils import get_openai_client

def upload_finetuning_file(file_path: Path) -> str:
    client = get_openai_client()
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="fine-tune")
    print(f"Fine-tuning file uploaded with ID: {file.id}")
    return file.id

def create_finetuning_job(
    training_file_id: str,
    validation_file_id: str,
    model: str,
    suffix: str
) -> str:
    client = get_openai_client()
    
    job_params = {
        "training_file": training_file_id,
        "validation_file": validation_file_id,
        "model": model,
        "suffix": suffix
    }
    
    job = client.fine_tuning.jobs.create(**job_params)
    print(f"Fine-tuning job created with ID: {job.id}")
    return job.id

def prepare_and_start_finetuning(
    training_file_path: Path,
    validation_file_path: Path,
    model: str,
    suffix: str
) -> str:
    training_file_id = upload_finetuning_file(training_file_path)
    
    validation_file_id = upload_finetuning_file(validation_file_path)
    
    job_id = create_finetuning_job(
        training_file_id,
        validation_file_id,
        model,
        suffix
    )
    
    return job_id