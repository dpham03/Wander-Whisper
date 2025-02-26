from datasets import load_from_disk

# Load the tokenized dataset
dataset_path = "./synthetic_prompts/tokenized_synthetic_travel_data"
dataset = load_from_disk(dataset_path)

# Search for the specific prompt
validation_data = dataset["validation"]

for i in range(5):  # Check first 5 entries
    print(f"✅ Sample {i}:")
    print("Prompt:", validation_data[i]["prompt"])
    print("\n")
    print("Expected Output:", validation_data[i]["output"])
    print("-----")
