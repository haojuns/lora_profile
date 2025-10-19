import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
import numpy as np
import evaluate
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# --- 1. Set up device (GPU if available, else CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Define model and tokenizer ---
model_name = "/data/user24161585/fc/ZZM_lora_teaching/deberta-v3-small-local"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# --- 3. Prepare dataset (AG News) ---
data_files = {
    "train": "/data/user24161585/fc/DATASET/AG_NEWS/train-00000-of-00001.parquet",
    "validation": "/data/user24161585/fc/DATASET/AG_NEWS/test-00000-of-00001.parquet"
}
full_dataset = load_dataset("parquet", data_files=data_files)


# --- NEW: Reduce dataset size to 1/20th ---
print("\nReducing dataset size...")
original_train_size = len(full_dataset["train"])
original_validation_size = len(full_dataset["validation"])

# Calculate 1/20th of the size
new_train_size = int(original_train_size / 20)
new_validation_size = int(original_validation_size / 20)

# Select the subset using .select()
train_subset = full_dataset["train"].select(range(new_train_size))
validation_subset = full_dataset["validation"].select(range(new_validation_size))

# Create a new DatasetDict with the subsets
dataset = DatasetDict({
    'train': train_subset,
    'validation': validation_subset
})

print(f"Original train samples: {original_train_size} -> New train samples: {len(dataset['train'])}")
print(f"Original validation samples: {original_validation_size} -> New validation samples: {len(dataset['validation'])}\n")


def preprocess_function(examples):
    # This tokenization will now run on the smaller dataset
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 4. Load base model ---
num_labels = 4
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    label2id={"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
    ignore_mismatched_sizes=True
)

# --- Move the base model to the selected device (GPU) ---
base_model.to(device)
print(f"Model loaded onto {next(base_model.parameters()).device}")


# Freeze base model parameters
for param in base_model.parameters():
    param.requires_grad = False

# --- 5. Configure LoRA (PEFT) ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_proj", "key_proj", "value_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# --- 6. Create PEFT model ---
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# --- 7. Define metrics ---
# MODIFIED: Load the metric from a local file to avoid network issues.
# Make sure 'accuracy_metric.py' is in the same directory.
metric = evaluate.load("/data/user24161585/fc/ZZM_lora_teaching/metrics/accuracy/accuracy.py")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# --- 8. Set up training ---
training_args = TrainingArguments(
    output_dir="/data/user24161585/fc/ZZM_lora_teaching/results/lora-deberta-finetuned-agnews",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy="steps",
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=20,
    eval_steps=20,
    save_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# --- 9. Create Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 10. Start training ---
print("Starting LoRA fine-tuning on AG News (small dataset)...")
trainer.train()

# --- 11. Save model ---
save_path = "/data/user24161585/fc/ZZM_lora_teaching/results/my-lora-adapter-agnews-small"
model.save_pretrained(save_path)
print(f"LoRA adapter (best model) has been saved to {save_path}")

