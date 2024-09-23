import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Setting up the model and processor paths
model_directory = "./qwen2vl_2b_instruct_lora_merged"

# Loading the fine-tuned model from the specified directory
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_directory,
    torch_dtype="auto",  # Automatically selects the appropriate torch data type
    device_map="auto"    # Automatically maps the model to the available GPU/CPU
).to('cuda')

# Loading the processor from the same directory as the model
processor = AutoProcessor.from_pretrained(model_directory)

# Load the CSV containing image URLs and metadata
csv_path = 'test.csv'  # Local path to the CSV file
df = pd.read_csv(csv_path)

# Function to download an image from a URL and resize it
def download_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((512, 512))  # Resize image to ensure model compatibility
        return img
    except Exception as e:
        return None

# Function to process a single row
def process_row(row):
    image_link = row['image_link']
    entity_name = row['entity_name']
    group_id = row['group_id']

    image = download_image(image_link)
    if image:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": f"Extract only {entity_name} value and units from the image."}]
        }]

        # Creating input for the model using the processor
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

        # Model inference
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)

        # Decode the model's output
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        extracted_text = output_text[0] if output_text else ""
        return {
            "image_link": image_link,
            "entity_name": entity_name,
            "group_id": group_id,
            "extracted_answer": extracted_text
        }
    else:
        return {
            "image_link": image_link,
            "entity_name": entity_name,
            "group_id": group_id,
            "extracted_answer": "Some error occured so nothing found"
        }

num_workers = int(os.cpu_count() * 0.8)  # Use 80% of CPU cores
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_row, row) for index, row in df.iterrows()]
    results = [future.result() for future in as_completed(futures)]

result_df = pd.DataFrame(results)
output_csv_path = 'extracted_results_qwen2.csv'
result_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}\n")
