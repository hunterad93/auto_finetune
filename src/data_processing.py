import json
from pathlib import Path
from typing import List, Dict, Any

def prepare_finetuning_data(
    batch_input_path: Path,
    batch_output_path: Path,
    output_dir: Path = Path("data/processed/finetuningJSONLs"),
    output_filename: str = None
) -> Path:
    """
    Prepare fine-tuning data by combining batch inputs and outputs.
    
    Args:
    batch_input_path: Path to the batch input JSONL file
    batch_output_path: Path to the batch output JSONL file
    output_dir: Directory to save the fine-tuning JSONL file
    output_filename: Optional filename for the output file
    
    Returns:
    Path to the created fine-tuning JSONL file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read batch input and output
    with open(batch_input_path, 'r') as f_in, open(batch_output_path, 'r') as f_out:
        batch_inputs = [json.loads(line) for line in f_in]
        batch_outputs = [json.loads(line) for line in f_out]
    
    # Combine inputs and outputs
    finetuning_data = []
    for input_item, output_item in zip(batch_inputs, batch_outputs):
        messages = input_item['body']['messages']
        system_message = next(msg['content'] for msg in messages if msg['role'] == 'system')
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')
        
        # Get the assistant's message from the output, preserving the JSON structure
        assistant_message = output_item['response']['body']['choices'][0]['message']['content']
        
        finetuning_item = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        finetuning_data.append(finetuning_item)
    
    # Save fine-tuning data
    if output_filename:
        output_file = output_dir / output_filename
    else:
        output_file = output_dir / f"finetuning_data_{batch_input_path.stem}.jsonl"
    
    with open(output_file, 'w') as f:
        for item in finetuning_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Fine-tuning data saved to: {output_file}")
    return output_file

def validate_finetuning_data(file_path: Path) -> bool:
    """
    Validate the structure of the fine-tuning data file.
    
    Args:
    file_path: Path to the fine-tuning JSONL file
    
    Returns:
    True if the file is valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                if not isinstance(item, dict):
                    print(f"Error: Item {i} is not a dictionary")
                    return False
                if list(item.keys()) != ['messages']:
                    print(f"Error: Item {i} should only have a 'messages' key")
                    return False
                if not isinstance(item['messages'], list) or len(item['messages']) != 3:
                    print(f"Error: Item {i} should have exactly 3 messages")
                    return False
                roles = [msg['role'] for msg in item['messages']]
                if roles != ['system', 'user', 'assistant']:
                    print(f"Error: Item {i} has incorrect message roles: {roles}")
                    return False
                for j, msg in enumerate(item['messages']):
                    if set(msg.keys()) != {'role', 'content'}:
                        print(f"Error: Message {j} in item {i} has incorrect keys")
                        return False
                    if not isinstance(msg['role'], str) or not isinstance(msg['content'], str):
                        print(f"Error: Role or content in message {j} of item {i} is not a string")
                        return False
                    if not msg['content'].strip():
                        print(f"Error: Content in message {j} of item {i} is empty or only whitespace")
                        return False
                # Validate that assistant's message is valid JSON
                try:
                    json.loads(item['messages'][2]['content'])
                except json.JSONDecodeError:
                    print(f"Error: Assistant's message in item {i} is not valid JSON")
                    return False
        return True
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file")
        return False
    except Exception as e:
        print(f"Error validating file: {str(e)}")
        return False