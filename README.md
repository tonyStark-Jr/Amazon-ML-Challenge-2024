
# Amazon ML Challenge - Team Tensorr

This repository contains the solution submitted by Team Tensorr for the **Amazon ML Challenge**. Our approach focuses on optimized data extraction from a dataset consisting of both images and textual data using state-of-the-art machine learning techniques.

## Project Overview

Our solution is divided into two main phases:

1. **Model Fine-Tuning**: We fine-tuned the Qwen-2-VL-2B-Instruct model on Amazon's provided training dataset to improve its text extraction capabilities from images.
2. **Data Processing and Completion**: We employed multi-threaded processing for efficient data handling and used the LLama-3.1-8B model to fill in gaps detected in the initial extraction phase.

## Approach

1. **Model Fine-Tuning**:
   - The Qwen-2-VL-2B model was fine-tuned on Amazon's dataset for enhanced text recognition.
2. **Multi-threaded Processing**:
   - We implemented multi-threading to handle the dataset efficiently, processing multiple data batches simultaneously.
3. **Text Extraction**:
   - The fine-tuned model extracts text from images, which is then stored in CSV files for further processing.
4. **Data Completion**:
   - The LLama-3.1-8B model was used to fill in any missing or incomplete data.
5. **Post-Processing**:
   - We utilized regex-based post-processing to ensure the final output adheres to the required format.

## Technologies Used

- **Hugging Face Transformers** for model loading and fine-tuning
- **PyTorch** for model handling
- **Pandas** for data manipulation
- **PIL** for image preprocessing
- **Python Threading** for multi-threaded processing

## Team Members

- **Sumit Awasthi** 
- **Prakhar Shukla**
- **Akshay Waghmare**
- **Srijan Jain**

