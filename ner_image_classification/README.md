# NER + Image Classification Pipeline

A pipeline that takes a text message and an image as input, extracts an animal name from the text using a NER model, classifies the animal in the image, and returns a boolean indicating whether they match.

## Project Structure
```
ner_image_classification/
├── ner/
│   ├── train.py        # NER model training
│   ├── inference.py    # NER model inference
│   └── dataset.py      # Dataset generation
├── classification/
│   ├── train.py        # Image classification training
│   ├── inference.py    # Image classification inference
│   └── data/           # Animal image dataset
├── models/             # Saved trained models
├── pipeline.py         # End-to-end pipeline
├── eda.ipynb           # Exploratory Data Analysis of Animals10 dataset
├── README.md           # This file
└── requirements.txt    # Dependencies
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### NER Model
- **Base model:** `bert-base-uncased` (HuggingFace Transformers)
- **Task:** Token classification with `B-ANIMAL` / `O` labels
- **Trained weights:** [AnnaDee/ner_model](https://huggingface.co/AnnaDee/ner_model) → place in `models/ner_model/`

### Image Classification Model
- **Base model:** `MobileNetV2` pretrained on ImageNet (torchvision)
- **Task:** 10-class animal classification
- **Trained weights:** [AnnaDee/image_classification_model](https://huggingface.co/AnnaDee/image_classification_model) → place in `models/image_classification_model/model.pth`

## Usage

### NER Model
```bash
# Train
python ner/train.py --epochs 5 --lr 3e-5 --batch_size 16 --output_dir models/ner_model

# Inference
python ner/inference.py --text "There is a cat in the picture."
```

### Image Classification
```bash
# Train
python classification/train.py --epochs 10 --lr 0.001 --batch_size 32

# Inference
python classification/inference.py --image path/to/image.jpg
```

### Pipeline
```bash
python pipeline.py --text "There is a cow in the picture." --image path/to/image.jpg
```

## Dataset
[Animals10 dataset from Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

Download and place in `classification/data/raw-img/`

## Demo
```bash
jupyter notebook demo.ipynb
```