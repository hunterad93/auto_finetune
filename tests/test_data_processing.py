import pytest
from pathlib import Path
import json
from src.data_processing import prepare_finetuning_data, validate_finetuning_data

@pytest.fixture(scope="module")
def example_paths():
    return {
        'input': Path('data/raw/batch_inputs/example_input.jsonl'),
        'output': Path('data/raw/batch_outputs/example_output.jsonl'),
        'processed': Path('data/processed/finetuningJSONLs')
    }

@pytest.fixture(scope="module")
def finetuning_files(example_paths):
    try:
        # Prepare the fine-tuning data
        train_file, test_file = prepare_finetuning_data(
            example_paths['input'],
            example_paths['output'],
            output_dir=example_paths['processed'],
            output_filename_prefix="example_finetune"
        )
    except Exception as e:
        pytest.fail(f"prepare_finetuning_data raised an exception: {e}")
    
    return train_file, test_file

def test_prepare_finetuning_data(example_paths, finetuning_files):
    train_file, test_file = finetuning_files
    
    # Check if the files were created
    assert train_file.exists(), f"Training file was not created at {train_file}"
    assert test_file.exists(), f"Testing file was not created at {test_file}"
    
    # Validate the structure of the created files
    assert validate_finetuning_data(train_file), f"Created training file at {train_file} is not valid"
    assert validate_finetuning_data(test_file), f"Created testing file at {test_file} is not valid"
    
    # Check the content of the created files
    train_data = read_jsonl(train_file)
    test_data = read_jsonl(test_file)
    
    # Assert we have the correct number of items (80% in train, 20% in test)
    total_items = len(train_data) + len(test_data)
    assert len(train_data) == pytest.approx(total_items * 0.8, abs=1), f"Expected ~80% of items in train set, but got {len(train_data)}/{total_items}"
    assert len(test_data) == pytest.approx(total_items * 0.2, abs=1), f"Expected ~20% of items in test set, but got {len(test_data)}/{total_items}"
    
    # Check the structure of each item in both files
    for dataset in [train_data, test_data]:
        for i, item in enumerate(dataset):
            validate_item_structure(item, i)
    
    print(f"Fine-tuning data successfully created and validated at: {train_file} and {test_file}")

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def validate_item_structure(item, index):
    assert isinstance(item, dict), f"Item {index} is not a dictionary"
    assert list(item.keys()) == ['messages'], f"Item {index} should only have a 'messages' key, but has keys: {list(item.keys())}"
    assert isinstance(item['messages'], list), f"Item {index}'s 'messages' value is not a list"
    assert len(item['messages']) == 3, f"Item {index} should have 3 messages, but has {len(item['messages'])}"
    
    roles = [msg['role'] for msg in item['messages']]
    assert roles == ['system', 'user', 'assistant'], f"Item {index} has incorrect message roles: {roles}"
    
    for j, msg in enumerate(item['messages']):
        assert set(msg.keys()) == {'role', 'content'}, f"Message {j} in item {index} has incorrect keys: {set(msg.keys())}"
        assert isinstance(msg['role'], str), f"Role in message {j} of item {index} is not a string"
        assert isinstance(msg['content'], str), f"Content in message {j} of item {index} is not a string"
        assert msg['content'].strip(), f"Content in message {j} of item {index} is empty or only whitespace"
    
    # Validate that assistant's message is valid JSON
    try:
        json.loads(item['messages'][2]['content'])
    except json.JSONDecodeError:
        pytest.fail(f"Assistant's message in item {index} is not valid JSON")

def test_examine_finetuning_files(finetuning_files):
    train_file, test_file = finetuning_files
    
    print(f"\nExamining training file: {train_file}")
    with open(train_file, 'r') as f:
        train_content = f.read()
        print(train_content)

    print(f"\nExamining testing file: {test_file}")
    with open(test_file, 'r') as f:
        test_content = f.read()
        print(test_content)

    print("Fine-tuning files examination completed.")