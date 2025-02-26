from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

def train_t5():
    MODEL_NAME = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # ✅ Save tokenizer at the beginning before training starts
    tokenizer.save_pretrained("fine_tuned_models/t5_tokenizer")

    # ✅ Load dataset correctly
    dataset = load_from_disk("./synthetic_prompts/tokenized_synthetic_travel_data")

    # ✅ Explicitly reference train and validation splits
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    training_args = TrainingArguments(
        output_dir="fine_tuned_models",
        num_train_epochs=3,  
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # ✅ Save final trained model
    model.save_pretrained("fine_tuned_models/fine_tuned_t5_small_travel_3_epochs")

    print("✅ Model trained and saved! Tokenizer was already saved at the beginning.")

if __name__ == "__main__":
    train_t5()
