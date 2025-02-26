from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer
import json
from sklearn.model_selection import train_test_split

def tokenize_function(examples, tokenizer):
    target_text = [json.dumps(output) for output in examples["output"]]
    #print("Target text: ", target_text[:1])

    # Tokenize input
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize output
    labels = tokenizer(
        target_text,
        max_length=256,
        padding="max_length",
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    with open("./synthetic_prompts/expanded_synthetic_travel_data.json", "r") as f:
        raw_data = json.load(f)['data']

    
    print(raw_data[0])
    # Split into train (80%) and validation (20%)
    train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    


    # Tokenize both datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Save tokenized datasets
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})
    
    dataset_dict.save_to_disk("./synthetic_prompts/tokenized_synthetic_travel_data")

