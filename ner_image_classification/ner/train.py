from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse

from dataset import generate_ner_dataset, animals, templates

# labels
label_list = ["O", "B-ANIMAL"]
label_to_id = {label: i for i, label in enumerate(label_list)} # {'O': 0, 'B-ANIMAL': 1}
id_to_label = {i: label for label, i in label_to_id.items()} # {0: 'O', 1: 'B-ANIMAL'}

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# function to tokenize and align labels
def tokenize_and_align_labels(tokens, labels):
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,  # already split into words
        truncation = True
    )

    aligned_labels = []
    word_ids = tokenized.word_ids()  # which token belongs to which word
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None: # special tokens [CLS], [SEP]
            aligned_labels.append(-100)
        elif word_id != previous_word_id: # first token of the word
            aligned_labels.append(label_to_id[labels[word_id]]) # id → "B-ANIMAL" → 1
        else: # subword (##w)
            aligned_labels.append(-100)
        previous_word_id = word_id

    return tokenized, aligned_labels

# Class for the dataset
class NERDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokenized, aligned_labels = tokenize_and_align_labels(
            sample["tokens"], # ['There', 'is', 'a', 'cow', 'in', 'the', 'picture']
            sample["labels"] # ['O', 'O', 'O', 'B-ANIMAL', 'O', 'O', 'O']
        )
        return {
            "input_ids": torch.tensor(tokenized["input_ids"]), # tensor([101, 2045, 2003, 1037, 10819, 102])
            "attention_mask": torch.tensor(tokenized["attention_mask"]), # tensor([1, 1, 1, 1, 1, 1])
            "labels": torch.tensor(aligned_labels) # tensor([-100, 0, 0, 1, 0, 0, 0, -100])
        }
    
# load model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=len(label_list),
    id2label = id_to_label,
    label2id = label_to_id
)

# create dataset
dataset = generate_ner_dataset(animals, templates)

# split into train and test
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=29)

# create dataset objects
train_dataset = NERDataset(train_data)
val_dataset = NERDataset(val_data)

# create data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./models/ner_model")
    args = parser.parse_args()

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    # training setup
    num_epochs = args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train loop
    for epoch in range(num_epochs):
        model.train() # set model to training mode
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation loop
        model.eval() # set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_val_loss += outputs.loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}")

    # save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)