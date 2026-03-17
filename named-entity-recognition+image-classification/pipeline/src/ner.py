import sys
from pathlib import Path
import torch
import string
import inflect

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from transformers import BertTokenizerFast, BertForTokenClassification
from pipeline.src.config import NER_MODEL_DIR, DEVICE

# Load the tokenizer and NER model
tokenizer = BertTokenizerFast.from_pretrained(str(NER_MODEL_DIR))
ner_model = BertForTokenClassification.from_pretrained(str(NER_MODEL_DIR))
ner_model.to(DEVICE)
ner_model.eval()

id2label = ner_model.config.id2label
_inflect = inflect.engine()

def extract_animal_entities(text: str) -> list:
    """
    Extract animal entities from the input text using the loaded NER model.
    
    Args:
        text (str): Input text.
    
    Returns:
        list: List of extracted animal entities.
    """
    words = text.split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    word_ids = encoding.word_ids(batch_index=0)
    inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = ner_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    
    entities = []
    current_entity = []
    previous_word_id = None
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        
        if word_id != previous_word_id:
            label = id2label[predictions[idx]]
            clean_word = words[word_id].strip(string.punctuation)
            if label == "B-ANIMAL":
                singular_word = _inflect.singular_noun(clean_word)
                if singular_word:
                    clean_word = singular_word
                if current_entity and word_id == previous_word_id + 1:
                    current_entity.append(clean_word)
                else:
                    if current_entity:
                        entities.append(" ".join(current_entity))
                    current_entity = [clean_word]
            else:
                if current_entity:
                    entities.append(" ".join(current_entity))
                    current_entity = []
        previous_word_id = word_id
    
    if current_entity:
        entities.append(" ".join(current_entity))
    
    return entities
