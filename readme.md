# LoRA Finetuning & GPU Memory Monitoring Script

This script finetunes a DeBERTa model on the AG News dataset using LoRA.

It features a custom `MonitoredTrainer` that tracks and plots GPU memory usage during the initial training iterations. This is highly useful for analyzing memory spikes and debugging Out-of-Memory (OOM) errors.

## Core Features

- **LoRA Finetuning**: Efficiently tunes the model using the `peft` library.
- **Configurable LoRA**: Choose between `'full'` (all target modules) or `'last_layers'` (only the last N layers).
- **GPU Memory Monitoring**: The custom `MonitoredTrainer` plots memory usage for the first few training iterations.
- **Dataset Subsetting**: Use `DATASET_REDUCTION_FACTOR` for quick debugging on a small subset of data.

## 1\. Setup & Environment

### 1.1 Download Required Files

1.  **Model:** Download the model files (e.g., `config.json`, `pytorch_model.bin`, etc.) from [Hugging Face - tasksource/deberta-small-long-nli](https://huggingface.co/tasksource/deberta-small-long-nli) and place them in your local `MODEL_NAME` directory.
2.  **Dataset:** Download the `.parquet` training and test files from [Hugging Face - fancyzhx/ag_news](https://www.google.com/search?q=https://huggingface.co/datasets/fancyzhx/ag_news).
3.  **Metric:** Download the `metric` folder from [GitHub - huggingface/evaluate](https://github.com/huggingface/evaluate/tree/main/metrics).

### 1.2 Install Dependencies

The script was tested with the following versions. You can install them via `pip`:

```bash
pip install datasets==3.2.0 evaluate==0.4.3 matplotlib==3.9.4 numpy==1.26.4 peft==0.7.0 torch==2.6.0 transformers==4.45.2
```

(Reference list from `pip list`)

```
datasets       3.2.0
evaluate       0.4.3
matplotlib     3.9.4
numpy          1.26.4
peft           0.7.0
torch          2.6.0
transformers   4.45.2
```

## 2\. Key Configuration (Modify Before Running)

Before running, you **MUST** modify the `CONFIGURATION` section at the top of the script, especially the paths:

- `MODEL_NAME`: Point this to your downloaded local DeBERTa model folder (e.g., `./sparse_lora/deberta-v3-small-local`).
- `TRAIN_FILE` & `VALIDATION_FILE`: Point these to your downloaded `.parquet` dataset files.
- `METRIC_PATH`: Point this to your downloaded local `accuracy` metric folder (e.g., `./sparse_lora/metrics/accuracy`).
- `OUTPUT_DIR` and `FINAL_ADAPTER_SAVE_PATH`: The directory to save results. Remember to create these empty folders.

: The path to save the final LoRA adapter.

Also, adjust other experimental parameters as needed:

- `LORA_MODE`: Choose `'full'` or `'last_layers'`.
- `DATASET_REDUCTION_FACTOR`: Set to `1` for a full training run, or a large value (e.g., `100`) for a quick debug run.
- `TARGET_GPU`: Set the GPU ID you wish to use (e.g., `"0"`).
- `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, etc.

The final file tree will be：

```plaintext
    DATASET
    └──AG_NEWS
    sparse_lora
    ├── deberta-v3-small-local
    │   ├── added_tokens.json
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── special_tokens_map.json
    │   ├── spm.model
    │   ├── tokenizer_config.json
    │   └── tokenizer.json
    ├── metrics
    │   └── accuracy
    │       └── ...
    ├── results
    │   └── ...
    ├── full_lora.py
    ├── readme.md
    └── sparse_lora.py
```

## 3\. How to Run

After completing the configuration, run the script from your terminal:

```bash
python sparse_lora.py
```

## 4\. Script Output

- **Training Checkpoints**: Saved in the `OUTPUT_DIR`.
- **Final LoRA Adapter**: Saved at the `FINAL_ADAPTER_SAVE_PATH` (includes `adapter_model.bin`, etc.).
- **Memory Usage Plot**: A `.png` plot saved in the `training_plots` subdirectory within your `OUTPUT_DIR`.
