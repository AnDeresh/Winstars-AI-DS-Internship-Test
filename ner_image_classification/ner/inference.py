from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import argparse
import os

# load model and tokenizer
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ner_model")
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def extract_animal(text):
    # tokenize input text
    tokenized = tokenizer(
        text,
        truncation = True, 
        return_tensors="pt"
    )

    # inference
    with torch.no_grad():
        outputs = model(**tokenized) # logits shape: (1, seq_len, num_labels)
        predictions = torch.argmax(outputs.logits, dim=2)[0] # shape: (seq_len,)
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
        labels = [model.config.id2label[pred.item()] for pred in predictions]
    
    # extract animal from labels
    animal = None
    for token, label in zip(tokens, labels):
        if label == "B-ANIMAL":
            animal = token
            break

    return animal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract animal from text")
    parser.add_argument("--text", type=str, help="Input text", required=True)
    args = parser.parse_args()

    animal = extract_animal(args.text)
    print(f"Extracted animal: {animal}")

