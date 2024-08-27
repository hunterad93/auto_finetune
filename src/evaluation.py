from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Type, Union, Tuple
from pydantic import BaseModel
from src.utils import get_openai_client
from src.batch_preparation import format_batch_request
from src.batch_processing import upload_batch_file, create_batch_job, wait_for_batch_completion, process_batch_results
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time


def monitor_finetuning_job(job_id: str, polling_interval: int = 60) -> str:
    """
    Monitor the fine-tuning job and return the fine-tuned model name when complete.

    Args:
    job_id: The ID of the fine-tuning job to monitor.
    polling_interval: Time in seconds between status checks.

    Returns:
    The name of the fine-tuned model.
    """
    client = get_openai_client()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        if status == "succeeded":
            print(f"Fine-tuning job {job_id} completed successfully.")
            return job.fine_tuned_model
        elif status in ["failed", "cancelled"]:
            raise Exception(f"Fine-tuning job {job_id} {status}.")
        
        print(f"Fine-tuning job status: {status}. Waiting {polling_interval} seconds...")
        time.sleep(polling_interval)

def prepare_evaluation_data(validation_file: Path) -> List[Dict[str, Any]]:
    """
    Prepare evaluation data from the validation file.
    """
    validation_data = []
    with open(validation_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            validation_data.append({
                "messages": [
                    msg for msg in item["messages"]
                    if msg["role"] in ["system", "user"]
                ]
            })
    return validation_data

def run_models_evaluation(
    evaluation_data: List[Dict[str, Any]],
    models: Dict[str, str],
    max_tokens: int,
    save_dir: Path,
    response_model: Optional[Type[BaseModel]] = None
) -> Dict[str, Path]:
    """
    Run evaluation for multiple models in a single batch and return the paths to results.
    """
    batch_requests = []
    custom_id_counter = 1
    for model_name, model in models.items():
        model_requests = format_batch_request(
            prompts=[item["messages"][-1]["content"] for item in evaluation_data],
            system_message=evaluation_data[0]["messages"][0]["content"],
            response_model=response_model,
            model=model,
            max_tokens=max_tokens
        )
        # Update custom_id for each request
        for request in model_requests:
            request["custom_id"] = f"request-{custom_id_counter}"
            custom_id_counter += 1
        batch_requests.extend(model_requests)

    # Save batch requests to a file
    input_file_path = save_dir / f"eval_input_all_models.jsonl"
    with open(input_file_path, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')

    # Upload batch file and create job
    file_id = upload_batch_file(input_file_path)
    time.sleep(5)  # Add a small delay to ensure the file is processed
    batch_id = create_batch_job(file_id)

    # Wait for job completion and process results
    wait_for_batch_completion(batch_id)
    results_path = process_batch_results(batch_id, save_dir, "eval_output_all_models")

    # Split the results into separate files for each model
    model_results = {}
    with open(results_path, 'r') as f:
        all_results = [json.loads(line) for line in f]

    for model_name in models.keys():
        model_output_path = save_dir / f"{model_name}_eval_output.jsonl"
        model_results[model_name] = model_output_path
        with open(model_output_path, 'w') as f:
            model_outputs = [result for result in all_results if result['request']['body']['model'] == models[model_name]]
            for output in model_outputs:
                f.write(json.dumps(output) + '\n')

    return model_results

def generate_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text."""
    openai_client = get_openai_client()
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
        encoding_format="float",
        dimensions=1024
    )
    return response.data[0].embedding

def compare_values(val1: Union[str, int, float], val2: Union[str, int, float]) -> Tuple[float, float]:
    """
    Compare two values based on their type.
    Returns a tuple of (string_similarity, numeric_similarity).
    """
    if isinstance(val1, str) and isinstance(val2, str):
        emb1 = generate_embedding(val1)
        emb2 = generate_embedding(val2)
        return cosine_similarity([emb1], [emb2])[0][0], np.nan
    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        max_val = max(abs(val1), abs(val2))
        similarity = 1 - (abs(val1 - val2) / max_val) if max_val != 0 else 1
        return np.nan, similarity
    else:
        return np.nan, np.nan

def compare_model_outputs(results: Dict[str, Path]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compare the outputs of different models, separating string and numeric comparisons.
    
    Args:
    results: Dictionary with model names as keys and paths to their output files as values.
    
    Returns:
    A dictionary of pairwise similarities between model outputs, with separate scores for strings and numbers.
    """
    model_outputs = {}
    for model, result_path in results.items():
        with open(result_path, 'r') as f:
            outputs = [json.loads(line)['response']['body']['choices'][0]['message']['content'] for line in f]
        model_outputs[model] = [json.loads(output) for output in outputs]  # Parse JSON strings

    similarities = {}
    models = list(results.keys())
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            string_similarities = []
            numeric_similarities = []
            
            for output1, output2 in zip(model_outputs[model1], model_outputs[model2]):
                for k in output1.keys():
                    string_sim, numeric_sim = compare_values(output1.get(k), output2.get(k))
                    if not np.isnan(string_sim):
                        string_similarities.append(string_sim)
                    if not np.isnan(numeric_sim):
                        numeric_similarities.append(numeric_sim)
            
            avg_string_sim = np.mean(string_similarities) if string_similarities else np.nan
            avg_numeric_sim = np.mean(numeric_similarities) if numeric_similarities else np.nan
            similarities[f"{model1}_vs_{model2}"] = (avg_string_sim, avg_numeric_sim)

    return similarities

# Update evaluate_models to return the new similarity format
def evaluate_models(
    validation_file: Path,
    finetuned_model: str,
    base_mini_model: str,
    large_model: str,
    max_tokens: int,
    save_dir: Path,
    response_model: Optional[Type[BaseModel]] = None
) -> Dict[str, Any]:
    """
    Evaluate finetuned mini model, base mini model, and large model on validation data.
    """
    evaluation_data = prepare_evaluation_data(validation_file)
    
    models = {
        "finetuned": finetuned_model,
        "base_mini": base_mini_model,
        "large": large_model
    }
    
    results = run_models_evaluation(
        evaluation_data,
        models,
        max_tokens,
        save_dir,
        response_model
    )
    
    # Compare model outputs
    similarities = compare_model_outputs(results)
    
    return {"results": results, "similarities": similarities}