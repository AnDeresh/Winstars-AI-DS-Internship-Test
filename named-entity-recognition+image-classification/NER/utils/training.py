import os
import torch
import pandas as pd
import numpy as np
import random
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, TrainerCallback, logging
from seqeval.metrics import precision_score, recall_score, f1_score
import nltk
from nltk.corpus import wordnet
import config

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def get_synonyms(word):
    """
    Retrieve synonyms from WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def augment_sentence(sentence, labels, aug_prob=0.3):
    """
    Augment a sentence by replacing non-entity words with their synonyms.
    """
    new_sentence = []
    for word, label in zip(sentence, labels):
        if label not in ["B-ANIMAL", "I-ANIMAL"] and random.random() < aug_prob:
            synonyms = get_synonyms(word)
            new_sentence.append(random.choice(synonyms) if synonyms else word)
        else:
            new_sentence.append(word)
    return new_sentence

def tokenize_and_align_labels(texts, labels, tokenizer, label2id, pad_label_id=-100):
    """
    Tokenize texts and align BIO labels with tokenized outputs.
    """
    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        return_offsets_mapping=True,
        padding=True
    )
    offset_mapping = tokenized_inputs.pop("offset_mapping")
    
    all_labels_aligned = []
    for i in range(len(tokenized_inputs.encodings)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(pad_label_id)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels[i]):
                    label_ids.append(label2id[labels[i][word_idx]])
                else:
                    label_ids.append(pad_label_id)
            else:
                if word_idx < len(labels[i]):
                    label_ids.append(label2id[labels[i][word_idx]])
                else:
                    label_ids.append(pad_label_id)
            previous_word_idx = word_idx
        all_labels_aligned.append(label_ids)
    
    tokenized_inputs["labels"] = all_labels_aligned
    return tokenized_inputs

class NERDataset(Dataset):
    """
    Custom Dataset class for NER.
    """
    def __init__(self, encodings):
        if isinstance(encodings["input_ids"][0], int):
            for key in encodings.keys():
                encodings[key] = [encodings[key]]
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

def main():
    # Load preprocessed BIO dataset CSV
    df = pd.read_csv(config.BIO_DATASET_CSV)
    df["Sentence"] = df["Sentence"].apply(lambda x: x.split())
    df["BIO Tags"] = df["BIO Tags"].apply(lambda x: x.split())

    sentences = df["Sentence"].tolist()
    labels_seq = df["BIO Tags"].tolist()

    print("Total dataset size:", len(sentences))

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sentences, labels_seq, test_size=0.2, random_state=42
    )

    # Augment training data
    aug_train_texts = []
    aug_train_labels = []
    for sent, lab in tqdm(zip(train_texts, train_labels), total=len(train_texts), desc="Augmenting training data"):
        for _ in range(2):
            aug_sent = augment_sentence(sent, lab, aug_prob=0.3)
            aug_train_texts.append(aug_sent)
            aug_train_labels.append(lab)

    train_texts += aug_train_texts
    train_labels += aug_train_labels

    print("Train dataset size after augmentation:", len(train_texts))

    # Analyze label distribution
    all_labels = [label for seq in labels_seq for label in seq]
    unique_labels = sorted(list(set(all_labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Compute class weights for balanced training
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(unique_labels),
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    train_encodings = tokenize_and_align_labels(train_texts, train_labels, tokenizer, label2id)
    val_encodings = tokenize_and_align_labels(val_texts, val_labels, tokenizer, label2id)

    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    # Metrics computation function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        true_labels = []
        true_preds = []
        for i in range(len(labels)):
            cur_labels = []
            cur_preds = []
            for j in range(len(labels[i])):
                if labels[i][j] != -100:
                    cur_labels.append(id2label[labels[i][j]])
                    cur_preds.append(id2label[preds[i][j]])
            true_labels.append(cur_labels)
            true_preds.append(cur_preds)
        precision = precision_score(true_labels, true_preds)
        recall = recall_score(true_labels, true_preds)
        f1 = f1_score(true_labels, true_preds)
        return {"precision": precision, "recall": recall, "f1": f1}

    # Early stopping callback
    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, patience=8, min_delta=1e-5):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float("inf")
            self.counter = 0

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            val_loss = metrics.get("eval_loss")
            if val_loss is None:
                return
            if val_loss + self.min_delta < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                control.should_training_stop = True

    early_stopping = EarlyStoppingCallback()

    num_labels = len(unique_labels)
    model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=num_labels, id2label=id2label, label2id=label2id)
    logging.set_verbosity_warning()
    model.classifier.bias.data.zero_()
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)

    training_args = TrainingArguments(
        output_dir=config.MODEL_DIR,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_strategy="epoch",
        learning_rate=config.TRAINING_ARGS["learning_rate"],
        per_device_train_batch_size=config.TRAINING_ARGS["batch_size"],
        per_device_eval_batch_size=config.TRAINING_ARGS["batch_size"],
        num_train_epochs=config.TRAINING_ARGS["num_epochs"],
        weight_decay=config.TRAINING_ARGS["weight_decay"],
        warmup_steps=config.TRAINING_ARGS["warmup_steps"],
        save_total_limit=2,
        logging_dir="./logs",
        report_to="none",
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics
    )

    print("Training NER model...")
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained(config.MODEL_DIR, safe_serialization=False)
    tokenizer.save_pretrained(config.MODEL_DIR)
    print("NER model trained and saved successfully!")

if __name__ == "__main__":
    main()
