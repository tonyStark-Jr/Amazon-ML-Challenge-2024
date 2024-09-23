
# Team Tensorr- Amazon ML Challenge Submission

## Overview

This README provides instructions for setting up and deploying the LLaMA Factory project, designed to fine-tune a LLaMA model with LoRA adapters. The process covers environment setup, model training.

## Prerequisites

- **Python Version**: Python 3.8 or later

## Setup Instructions

### Step 1: Clone the Repository

Begin by cloning the LLaMA Factory repository and navigate into the directory:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### Step 2: Install Dependencies

Install all necessary Python packages to ensure compatibility and functionality:

```bash
pip install -r requirements.txt
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install -e ".[torch, metrics]"
pip install liger-kernel
```

### Step 3: Configure Environment

Set the environment variable for Gradio sharing:

```python
import os
os.environ["GRADIO_SHARE"] = "1"
```

### Step 4: Model Fine-Tuning

Set up the training parameters and initiate the fine-tuning process:

```python
import json

# Training configuration
args = {
    "stage": "sft",
    "do_train": True,
    "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
    "dataset": "amazon_ml_dataset",
    "template": "qwen2_vl",
    "finetuning_type": "lora",
    "output_dir": "qwen2vl_lora",
}

# Save configuration to JSON
json.dump(args, open("train_args_qwen2.json", "w", encoding="utf-8"), indent=2)

# Start training
!llamafactory-cli train train_args_qwen2.json
```

### Step 5: Export and Upload Model

After training, prepare the model for export and upload it to the Hugging Face Hub:

```python
# Export configuration
args = {
    "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
    "adapter_name_or_path": "qwen2vl_lora",
    "export_dir": "qwen2vl_2b_instruct_lora_merged",
}

# Save export configuration
json.dump(args, open("merge_args_qwen2.json", "w", encoding="utf-8"), indent=2)

# Export and upload
# !llamafactory-cli export merge_args_qwen2.json


### Step 6: Verification

Confirm the successful deployment of the model:

```python
print("Model pushed to: your_username/Qwen2-VL-2B-Instruct-LoRA-FT")
```

## Conclusion

This README outlines the essential steps for participating in the Amazon ML Challenge with the LLaMA Factory project. 
