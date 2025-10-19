import os
import torch
import torch.nn as nn
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
import re
import threading
import time
import queue
from datetime import datetime
import matplotlib.pyplot as plt

# ==============================================================================
# --- MEMORY MONITORING UTILITIES ---
# ==============================================================================

class MemoryMonitor:
    """
    A class to monitor GPU memory usage in a separate thread.
    """
    def __init__(self, interval=0.1):
        self.interval = interval
        self.memory_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.start_time = None
        self.monitor_thread = None

    def monitor_memory(self):
        """
        The target function for the monitoring thread. Records memory usage at intervals.
        """
        self.start_time = time.time()
        while not self.stop_flag.is_set():
            if torch.cuda.is_available():
                memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
                current_time = time.time() - self.start_time
                self.memory_queue.put((current_time, memory))
            time.sleep(self.interval)

    def start(self):
        """Starts the memory monitoring thread."""
        if not torch.cuda.is_available():
            print("CUDA not available. Memory monitoring will be skipped.")
            return
        self.stop_flag.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_memory)
        self.monitor_thread.start()
        print("Memory monitor started.")

    def stop(self):
        """Stops the memory monitoring thread."""
        if self.monitor_thread is None:
            return
        self.stop_flag.set()
        self.monitor_thread.join()
        print("Memory monitor stopped.")

    def get_measurements(self):
        """Retrieves all memory measurements from the queue."""
        measurements = []
        while not self.memory_queue.empty():
            measurements.append(self.memory_queue.get())
        return sorted(measurements)  # Sort by timestamp

class MonitoredTrainer(Trainer):
    """
    A custom Hugging Face Trainer that monitors and plots GPU memory usage.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_monitor = MemoryMonitor(interval=0.1)
        self.memory_measurements = []
        # Add attributes for tracking iterations
        self.num_iters_to_plot = 4
        self.iteration_end_times = []

    def train(self, *args, **kwargs):
        """
        Overrides the train method to start/stop memory monitoring and plot results.
        """
        # Reset iteration times at the start of training
        self.iteration_end_times = []
        self.memory_monitor.start()
        try:
            # Call the original train method from the parent Trainer class
            result = super().train(*args, **kwargs)
        finally:
            # Ensure the monitor is always stopped
            self.memory_monitor.stop()
            self.memory_measurements = self.memory_monitor.get_measurements()
        
        self.plot_memory_usage()
        return result

    def training_step(self, model, inputs):
        """
        Overrides the training_step to record the timestamp after the first few iterations.
        """
        # Perform the actual training step
        loss = super().training_step(model, inputs)

        # Record the timestamp after the step is complete (relative to the monitor's start time)
        if self.state.global_step <= self.num_iters_to_plot:
            # global_step is 1-indexed. len(iteration_end_times) is 0-indexed.
            # This ensures we record the time for each step only once.
            if len(self.iteration_end_times) < self.state.global_step:
                if self.memory_monitor.start_time is not None:
                    relative_time = time.time() - self.memory_monitor.start_time
                    self.iteration_end_times.append(relative_time)
        return loss

    def plot_memory_usage(self):
        """
        Plots the collected memory usage data for the first few iterations and saves it to a file.
        """
        if not self.memory_measurements:
            print("No memory measurements were recorded to plot.")
            return

        print("Plotting memory usage for the first few iterations...")
        times, memories = zip(*self.memory_measurements)
        
        max_time_to_plot = None
        # Check if we have recorded times for the iterations
        if self.iteration_end_times:
            # Use the latest recorded time as the cutoff
            max_time_to_plot = self.iteration_end_times[-1]

        # If we have a cutoff time, slice the data
        if max_time_to_plot is not None:
            end_index = len(times)
            for i, t in enumerate(times):
                # We add a small buffer (e.g., 0.5s) to ensure the full peak of the last iteration is visible
                if t > max_time_to_plot + 0.5:
                    end_index = i
                    break
            
            times_to_plot = times[:end_index]
            memories_to_plot = memories[:end_index]
            title = f'GPU Memory Usage During First {len(self.iteration_end_times)} Iterations'
        else:
            # Fallback if no iteration times were recorded for some reason
            print("Warning: Could not determine iteration end times. Plotting the first 5 seconds as a fallback.")
            max_time_to_plot = 5.0
            end_index = len(times)
            for i, t in enumerate(times):
                if t > max_time_to_plot:
                    end_index = i
                    break
            times_to_plot = times[:end_index]
            memories_to_plot = memories[:end_index]
            title = f'GPU Memory Usage During First {int(max_time_to_plot)} Seconds'

        if not times_to_plot:
            print("No memory measurements within the plotting window.")
            return

        plt.figure(figsize=(14, 7))
        plt.plot(times_to_plot, memories_to_plot, label="GPU Memory Usage")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Allocated Memory (MB)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Ensure the output directory for plots exists
        plot_dir = os.path.join(self.args.output_dir, "training_plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(plot_dir, f'memory_usage_{timestamp}.png')
        
        plt.savefig(save_path)
        print(f"Memory usage plot saved to: {save_path}")
        plt.close()

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# -- Run Mode --
# LoRA Fine-tuning Mode: 'full' or 'last_layers'
LORA_MODE = 'last_layers' 
LAYERS_TO_TRANSFORM = 2  # Only used if LORA_MODE is 'last_layers'

# -- Paths --
MODEL_NAME = "./sparse_lora/deberta-v3-small-local"
TRAIN_FILE = "./DATASET/AG_NEWS/train-00000-of-00001.parquet"
VALIDATION_FILE = "./DATASET/AG_NEWS/test-00000-of-00001.parquet"
METRIC_PATH = "./sparse_lora/metrics/accuracy/accuracy.py"
OUTPUT_DIR = "./sparse_lora/results/lora-deberta-finetuned-agnews"
FINAL_ADAPTER_SAVE_PATH = f"./sparse_lora/results/my-lora-adapter-agnews-small-{LORA_MODE}"

# -- Dataset Settings --
DATASET_REDUCTION_FACTOR = 100  # Use 1/20th of the data. Set to 1 for full dataset.

# -- Hardware Settings --
TARGET_GPU = "0" # Adjusted to a common default, change if needed

# -- Training Hyperparameters --
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01

# -- Logging, Evaluation, and Saving Frequency --
STEPS_FREQUENCY = 50

# -- LoRA Hyperparameters --
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# ==============================================================================
# --- SCRIPT START ---
# ==============================================================================

def main():
    # Set the specific GPU to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU

    # --- 1. Set up device (GPU if available, else CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Define model and tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- 3. Prepare dataset (AG News) ---
    data_files = {"train": TRAIN_FILE, "validation": VALIDATION_FILE}
    full_dataset = load_dataset("parquet", data_files=data_files)

    # --- Reduce dataset size ---
    if DATASET_REDUCTION_FACTOR > 1:
        print(f"\nReducing dataset size by a factor of {DATASET_REDUCTION_FACTOR}...")
        original_train_size = len(full_dataset["train"])
        original_validation_size = len(full_dataset["validation"])
        new_train_size = int(original_train_size / DATASET_REDUCTION_FACTOR)
        new_validation_size = int(original_validation_size / DATASET_REDUCTION_FACTOR)
        train_subset = full_dataset["train"].select(range(new_train_size))
        validation_subset = full_dataset["validation"].select(range(new_validation_size))
        dataset = DatasetDict({'train': train_subset, 'validation': validation_subset})
        print(f"Original train samples: {original_train_size} -> New train samples: {len(dataset['train'])}")
        print(f"Original validation samples: {original_validation_size} -> New validation samples: {len(dataset['validation'])}\n")
    else:
        dataset = full_dataset
        print("Using full dataset.\n")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 4. Load base model ---
    num_labels = 4
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        label2id={"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
        ignore_mismatched_sizes=True
    )
    base_model.to(device)
    print(f"Model loaded onto {next(base_model.parameters()).device}")

    # --- 5. Freeze base model parameters ---
    for param in base_model.parameters():
        param.requires_grad = False

    # --- 6. Configure LoRA based on LORA_MODE ---
    print(f"Configuring LoRA with mode: '{LORA_MODE}'")

    if LORA_MODE == 'full':
        target_modules_pattern = ["query_proj", "key_proj", "value_proj"]
        print("Applying LoRA to all targetable modules.\n")

    elif LORA_MODE == 'last_layers':
        num_layers = base_model.config.num_hidden_layers
        target_layer_indices = [num_layers - 1 - i for i in range(LAYERS_TO_TRANSFORM)]
        layer_pattern = "|".join([str(i) for i in target_layer_indices])
        target_modules_pattern = f".*deberta\\.encoder\\.layer\\.({layer_pattern})\\..*\\.(query_proj|key_proj|value_proj)"
        print(f"Total encoder layers: {num_layers}")
        print(f"Applying LoRA to layers: {target_layer_indices}")
        print(f"PEFT target_modules regex: {target_modules_pattern}\n")
        
    else:
        raise ValueError(f"Invalid LORA_MODE: '{LORA_MODE}'. Choose 'full' or 'last_layers'.")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules_pattern,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # --- 7. Create PEFT model ---
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # --- 8. Define metrics ---
    metric = evaluate.load(METRIC_PATH)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # --- 9. Set up training ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=STEPS_FREQUENCY,
        eval_steps=STEPS_FREQUENCY,
        save_steps=STEPS_FREQUENCY,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # --- 10. Create MonitoredTrainer ---
    trainer = MonitoredTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 11. Start training ---
    print(f"\nStarting LoRA fine-tuning (mode: {LORA_MODE})...")
    trainer.train()

    # --- 12. Save model ---
    model.save_pretrained(FINAL_ADAPTER_SAVE_PATH)
    print(f"\nLoRA adapter (best model) has been saved to {FINAL_ADAPTER_SAVE_PATH}")

if __name__ == "__main__":
    main()


