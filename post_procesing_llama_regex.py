# %%
# Ensure you have the necessary packages installed
!pip install unsloth datasets transformers torch

# %%
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer

# Model configuration
max_seq_length = 2048  # Maximum sequence length
dtype = None  # Auto detection of data type, e.g., Float16 or BFloat16 based on hardware
load_in_4bit = True  # Enable 4-bit quantization to reduce memory usage

# Initialize the model and tokenizer with the specified model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# %%
# Model tuning with PEFT (Positional Encoding Fourier Transform)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Tuning parameter for positional encoding
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",  # Optimizes VRAM usage
    random_state=3407
)

# %%
# Function to format prompts
def formatting_prompts_func(examples):
    texts = [
        f"### Instruction: {instr}\n### Text: {inp}\n### Response: {out}" + tokenizer.eos_token
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return {"text": texts}

# %%
# Entity to units mapping for various measurements
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}

# Function to generate responses based on the model
def generate_response(text, entity):
    prompt = f"Extract values related to {entity}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0])

# Example usage
# print(generate_response("The width of the table is 100 cm.", "width"))

# %%
# Function to process and update a DataFrame
def update_dataframe(data_path):
    data = pd.read_csv(data_path)
    for index, row in data.iterrows():
        if pd.isna(row['extracted_answer']):
            result = generate_response(row['extracted_answer'], row['entity_name'])
            data.at[index, 'extracted_result'] = result
    return data
    print("DataFrame updated and saved!")

# Example call to process local data
llama_out=update_dataframe("extracted_results_qwen2.csv")




import re
# Mapping of common variations of units to standardized units as defined previously
unit_conversion_map = {
    # Add all previous mappings here
    "cm": "centimetre", "meters": "metre", "mm": "millimetre",
    "in": "inch", "ft": "foot", "feet": "foot", "yd": "yard",
    "g": "gram", "kg": "kilogram", "mg": "milligram", "mcg": "microgram",
    "oz": "ounce", "lbs": "pound", "lb": "pound", "tonne": "ton",
    "kV": "kilovolt", "mV": "millivolt", "kW": "kilowatt", "W": "watt",
    "cl": "centilitre", "dl": "decilitre", "l": "litre", "ml": "millilitre"
}

# Standardization function as defined previously
def standardize_units(text, entity_type, entity_unit_map):
    matches = re.finditer(r"(\d+\.?\d*\s*)([a-zA-Z]+)", text)
    results = []
    for match in matches:
        value, unit = match.groups()
        unit = unit.lower()
        if unit in unit_conversion_map:
            unit = unit_conversion_map[unit]
        if unit in entity_unit_map[entity_type]:
            results.append(f"{value.strip()} {unit}")
    return "; ".join(results) if results else "No valid measurements found."


# Apply the function to each row in the DataFrame
llama_out['prediction'] = llama_out.apply(lambda row: standardize_units(row['extracted_result'], row['entity_name'], entity_unit_map), axis=1)
test_df=pd.read_csv("test.csv")
final_df=pd.DataFrame()

final_df["index"]=test_df["index"]

final_df["prediction"]=llama_out["prediction"]

final_df.to_csv("final_result.csv",index=False)