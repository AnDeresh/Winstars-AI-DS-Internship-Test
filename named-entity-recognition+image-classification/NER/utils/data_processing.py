import json
import re
import logging
from pathlib import Path
from typing import List, Tuple, Any

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

import config

# Download necessary NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def load_dataset(file_path: Path) -> List[Any]:
    """
    Load dataset from a JSON file.

    :param file_path: Path to the JSON file.
    :return: List of dataset entries.
    """
    logging.info(f"Loading dataset from {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def convert_to_bio(sentence: str, label: str) -> List[Tuple[str, str]]:
    """
    Convert a sentence to BIO format with lemmatization.

    :param sentence: The input sentence.
    :param label: The target label to match.
    :return: A list of tuples (word, BIO tag).
    """
    words = sentence.split()
    bio_tags = ["O"] * len(words)

    for i, word in enumerate(words):
        # Remove punctuation and lemmatize the word
        clean_word = re.sub(r"[^a-zA-Z]", "", word)
        lemma = lemmatizer.lemmatize(clean_word.lower())

        if lemma == label.lower():
            bio_tags[i] = "B-ANIMAL"

    return list(zip(words, bio_tags))


def bio_to_dataframe(bio_data: List[List[Tuple[str, str]]]) -> pd.DataFrame:
    """
    Convert BIO dataset to a pandas DataFrame.

    :param bio_data: List of BIO formatted sentences.
    :return: DataFrame with sentences and their corresponding BIO tags.
    """
    sentences = []
    tags = []
    for sentence in bio_data:
        words, bio_tags = zip(*sentence)
        sentences.append(" ".join(words))
        tags.append(" ".join(bio_tags))
    return pd.DataFrame({"Sentence": sentences, "BIO Tags": tags})


def process_dataset(input_file: Path, output_json: Path, output_csv: Path) -> None:
    """
    Process dataset into BIO format and save to JSON and CSV.

    :param input_file: Path to the raw dataset JSON file.
    :param output_json: Path to save the processed BIO JSON.
    :param output_csv: Path to save the processed CSV file.
    """
    data = load_dataset(input_file)
    bio_data_fixed = [convert_to_bio(entry["text"], entry["label"]) for entry in data]

    # Save BIO dataset to JSON
    logging.info(f"Saving BIO dataset to JSON at {output_json}")
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(bio_data_fixed, f, indent=4)

    # Convert to DataFrame and save to CSV
    df = bio_to_dataframe(bio_data_fixed)
    logging.info(f"Saving BIO dataset to CSV at {output_csv}")
    df.to_csv(output_csv, index=False)
    logging.info(f"BIO dataset saved successfully as {output_csv}")


if __name__ == "__main__":
    # Use Path objects for file paths
    input_path = Path(config.NER_DATASET_JSON)
    output_json_path = Path(config.BIO_DATASET_JSON)
    output_csv_path = Path(config.BIO_DATASET_CSV)

    process_dataset(input_path, output_json_path, output_csv_path)