import logging
import string
from typing import List

import inflect
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set the device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


class AnimalNER:
    """
    A class to handle loading a pre-trained BERT model for animal NER inference,
    tokenizing input text, and extracting animal entities.
    """

    def __init__(self, model_dir: str) -> None:
        """
        Initialize the AnimalNER class by loading the tokenizer and model from the given directory.

        :param model_dir: Directory path where the trained model and tokenizer are saved.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model = BertForTokenClassification.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()

        # Retrieve the id2label mapping from the model configuration
        self.id2label = self.model.config.id2label

        # Initialize inflect engine for converting plural to singular
        self.inflect_engine = inflect.engine()

    def extract_animal_entities(self, text: str) -> List[str]:
        """
        Extract animal entities from the input text using a pre-trained model.

        :param text: The input text from which to extract animal entities.
        :return: A list of extracted animal entity strings.
        """
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        word_ids = encoding.word_ids(batch_index=0)
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        entities = []
        current_entity = []
        previous_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            # We only process when we move to a new word index
            if word_id != previous_word_id:
                label = self.id2label[predictions[idx]]
                clean_word = words[word_id].strip(string.punctuation)

                if label == "B-ANIMAL":
                    # Convert plural to singular if possible
                    singular_word = self.inflect_engine.singular_noun(clean_word)
                    if singular_word:
                        clean_word = singular_word

                    # If consecutive to the previous word, keep building
                    if current_entity and word_id == previous_word_id + 1:
                        current_entity.append(clean_word)
                    else:
                        # If we had a previous entity, append it before starting a new one
                        if current_entity:
                            entities.append(" ".join(current_entity))
                        current_entity = [clean_word]
                else:
                    # If we're no longer in an ANIMAL tag, finalize the current entity
                    if current_entity:
                        entities.append(" ".join(current_entity))
                        current_entity = []

            previous_word_id = word_id

        # Append any remaining entity
        if current_entity:
            entities.append(" ".join(current_entity))

        return entities

    def display_examples(self, sentences: List[str]) -> None:
        """
        Display each sentence and its extracted animal entities using logging.

        :param sentences: A list of sentences for entity extraction.
        """
        for sentence in sentences:
            animals = self.extract_animal_entities(sentence)
            logging.info(f"Sentence: {sentence}")
            logging.info(f"Extracted Animals: {', '.join(animals) if animals else 'None'}")
            logging.info("-" * 80)


def main() -> None:
    """
    Main function to demonstrate the usage of AnimalNER with sample sentences.
    """
    test_cases = [
        "A lion roared in the savannah.",
        "Two tigers were lurking in the jungle.",
        "A group of penguins waddled on the ice.",
        "I spotted a zebra near the waterhole.",
        "A herd of elephants crossed the road."
    ]

    # Initialize the AnimalNER with the path to your model directory
    ner_model = AnimalNER(config.MODEL_DIR)
    ner_model.display_examples(test_cases)


if __name__ == "__main__":
    main()