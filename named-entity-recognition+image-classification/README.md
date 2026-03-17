# Named entity recognition + image classification

**Named entity recognition + image classification** is a multi-component project that combines image classification and named-entity recognition (NER) to verify whether a given text mentions the same animal depicted in an image. The project is divided into three main parts:

1. **Image Classification** – for training and inference on animal images.
2. **Named Entity Recognition (NER)** – for extracting animal entities from text.
3. **ML Pipeline** – for integrating both models into a unified workflow.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [Image Classification](#image-classification)
  - [NER](#ner)
  - [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Model](#model)
- [Demo Notebook](#demo-notebook)
- [Key Scripts](#key-scripts)

---

## Project Overview

This project aims to:

1. **Train and run an image classifier** that recognizes animals in images.  
2. **Train and run a NER model** that extracts animal entities from text.  
3. **Combine both** to check if the text mentions the animal predicted from the image.

Example scenario:
- Text: `"There is a cat sitting on the grass."`
- Image: An image with a cat.
- The pipeline verifies whether the predicted animal from the image (`"cat"`) is found in the text.

---

## Directory Structure

A high-level overview of key folders and files:

```
ANIMALVERIFY/
├── image_classification/
│   ├── data/
│   │   ├── raw-img/                # Raw input images for classification
│   │   └── processed-img/          # Preprocessed images ready for training/inference
│   ├── notebooks/                  # Jupyter notebooks for experiments and analysis
│   ├── scripts/
│   │   ├── run_training.py         # Script to train the image classifier
│   │   └── run_inference.py        # Script to run inference with the image classifier
│   └── translate.py                # (Optional) Additional script for data translation or augmentation
├── NER/
│   ├── data/                       # Data for training/testing the NER model
│   ├── model/                      # Pre-trained NER model files
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── notebooks/                  # Jupyter notebooks for NER experiments
│   └── utils/                      # Utility scripts for the NER model
│       ├── run_inference.py        # Script for NER inference
│       └── run_training.py         # Script to train the NER model
├── pipeline/
│   ├── evaluation_dataset/         # Evaluation dataset for pipeline testing
│   │   ├── evaluation_data.json    # JSON file containing evaluation test cases
│   │   └── evaluation_images/      # Folder containing evaluation images
│   ├── notebook/                   # Notebooks demonstrating pipeline usage
│   ├── scripts/
│   │   └── main.py                 # Main script for running the integrated pipeline
│   └── src/                        # Source code for the pipeline
│       ├── config.py               # Centralized configuration for paths and settings
│       ├── evaluation.py           # Pipeline evaluation functions and interactive mode
│       ├── image_classifier.py     # Functions for loading the classifier and predicting images
│       ├── ner.py                  # Functions for loading the NER model and extracting entities
│       ├── pipeline.py             # Main pipeline integration of NER and image classification
│       └── __init__.py             # Makes the src folder a Python package
├── requirements.txt                # Python dependencies
└── .gitignore
```


### Folders

- **`image_classification/`**  
  Contains code and data for training and running an animal image classifier.  

- **`NER/`**  
  Contains code and data for training and running a Named Entity Recognition model to extract animals from text.  

- **`pipeline/`**  
  Combines the outputs of both the classifier and the NER model. This is where you’ll find the final pipeline code (`pipeline.py`) that checks whether the predicted animal from the image matches the extracted entity from text.

### Notable Files

- **`run_training.py`, `run_inference.py`**: Scripts to train and test models in both `image_classification/` and `NER/`.
- **`pipeline/src/config.py`**: Centralized configuration for file paths, device settings, etc.
- **`pipeline/src/pipeline.py`**: Main logic that calls the NER and image classifier to verify if the text and image match.
- **`pipeline/src/evaluation.py`**: Contains functions for automatic testing (`test_evaluation_data`) and an interactive mode.

---

## Installation and Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/AnimalVerify.git
   cd AnimalVerify
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```  

3. **Install dependencies (`requirements.txt`)**:

   ```bash
   pip install -r requirements.txt
   ```  

## Usage

### Image Classification

#### Training
Navigate to the `image_classification/scripts/` directory (or run via the `-m` flag) and execute:
```bash
python run_training.py
```
This command trains the model on preprocessed image dataset.

### Inference

**Image Classification Inference**

To perform inference on a single image, run:
```bash
python run_inference.py --image path/to/image.jpg
```
This will output the predicted animal class for the provided image.

### NER Inference

To extract animal entities from a text snippet, run:
```bash
python ner_inference.py --text "I saw a dog in the park."
```
This command will output the recognized animal entities from the text.

### Pipeline

The pipeline integrates both image classification and NER into a single workflow. It provides two main entry points:

- **test_evaluation_data()**: Runs automated tests based on `evaluation_data.json`.
- **interactive_mode()**: Allows manual input of text and an image path.

**Running via CLI**

From the project root (`AnimalVerify/`), run:
```bash
python -m pipeline.src.evaluation
```

This command executes `test_evaluation_data()` and then enters interactive mode if no `--text` or `--image` arguments are provided.

Alternatively, to directly pass text and image arguments to the pipeline, run:
```bash
python -m pipeline.scripts.main --text "A cat on the grass" --image "path/to/cat.jpg"
```

## Dataset

Due to their large size, the image datasets are not tracked by Git. Instead, you can download the required data externally. I use the **Animals10** dataset available on Kaggle. 

Download the dataset from:  
[Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

Once downloaded, place the files in the following directories:

- Raw images should be placed in:  
  `AnimalVerify/Image_Classification/data/raw-img`
  
- Processed images will be generated and saved in:  
  `AnimalVerify/Image_Classification/data/processed-img`
  
- Augmented images will be stored in:  
  `AnimalVerify/Image_Classification/data/augmented-img`

After placing the raw images, run the preprocessing script (`run_preprocessing.py`) to generate the processed images, and then run the dataset preparation script (`prepare_dataset.py`) if needed.

## Model

Due to GitHub's file size limits, the NER model (pytorch_model.bin) is not included in this repository. You can download the model from Huggingface:

[Download NER Model from Huggingface](https://huggingface.co/AnnaDee/NER_pytorch_model/blob/main/pytorch_model.bin)

After downloading, place the model files in the following directory:
```AnimalVerify/NER/model/```

## Demo Notebook

A demo Jupyter Notebook is provided in the `pipeline/notebook/demo_pipeline.ipynb`. This notebook contains examples demonstrating how the solution works and explains edge cases, such as:

- Handling cases where no animal entity is detected in the text.
- Verifying the consistency between the image prediction and text extraction.
- Using interactive mode for manual testing.

To open the notebook with Jupyter Notebook or JupyterLab, run:

```bash
jupyter notebook pipeline/notebook/demo_pipeline.ipynb
```

### Key Scripts

- **image_classification/scripts/run_training.py**: Trains the image classifier.
- **image_classification/scripts/run_inference.py**: Performs inference on a single image.
- **NER/utils/run_training.py**: Trains the NER model.
- **NER/utils/run_inference.py**: Performs inference on a text snippet.
- **pipeline/scripts/main.py**: Main entry point for the integrated pipeline (supports interactive mode and CLI arguments).
- **pipeline/src/evaluation.py**: Contains `test_evaluation_data()` and `interactive_mode()` for pipeline evaluation.
