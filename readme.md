# LoRA Parameter Freezing Experiment

This project provides a practical demonstration of **Low-Rank Adaptation (LoRA)** in a Transformer model. The primary focus is to analyze and compare the GPU memory usage under different parameter freezing strategies.

## 📋 Overview

This script performs the following actions:

  - **Implements LoRA**: A custom `lora.Linear` layer is created to seamlessly integrate LoRA capabilities into standard PyTorch models.
  - **Builds a Transformer**: A simple Transformer model is constructed using the LoRA-enabled layers for demonstration.
  - **Monitors Memory**: A `MemoryMonitor` class tracks real-time GPU memory allocation during the training loop.
  - **Runs Experiments**: The main script trains the Transformer model under several distinct conditions:
      - **All Frozen**: No training occurs.
      - **Only LoRA**: Only the `lora_A` and `lora_B` matrices are trainable.
      - **Block-Specific LoRA**: Only LoRA parameters within a specific Transformer block are trained.
      - **All Trainable**: All model parameters are fully trainable (traditional fine-tuning).
  - **Visualizes Results**: The script generates a plot comparing the memory consumption of each training strategy and saves it.

## 🚀 Usage

To run the experiment, ensure you have PyTorch and Matplotlib installed. Then, simply execute the Python script from your terminal.

```bash
python your_script_name.py
```

The script will output the number of trainable parameters for each condition and the training loss. After all experiments are complete, a plot named `memory_usage.png` will be saved in the `TaskMate/taskmate/playground/` directory.

## 📊 Experiment Analysis

This experiment visually demonstrates the core benefit of LoRA: **memory efficiency**.

By freezing the original model weights and only training the small, low-rank matrices, LoRA significantly reduces the number of trainable parameters. As shown in the output plot, this directly translates to lower peak GPU memory usage compared to full fine-tuning, making it possible to adapt large models on consumer-grade hardware.