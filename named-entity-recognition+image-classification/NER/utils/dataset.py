# dataset.py
# This module creates a NER dataset from Wikipedia using the Wikipedia API

import wikipediaapi
import re
import json
import pandas as pd
import random
import config

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyNERBot/1.0 (contact: myemail@example.com)', language='en')

# List of animals to extract articles for
animals = [
    "Dog", "Cat", "Elephant", "Horse", "Cow", "Sheep", "Goat", "Lion", "Tiger", "Bear",
    "Wolf", "Fox", "Deer", "Rabbit", "Kangaroo", "Panda", "Leopard", "Cheetah", "Giraffe", "Zebra",
    "Hippopotamus", "Rhinoceros", "Bison", "Buffalo", "Moose", "Squirrel", "Otter", "Beaver", "Raccoon", "Skunk",
    "Armadillo", "Anteater", "Sloth", "Opossum", "Chinchilla", "Ferret", "Hedgehog", "Bat", "Orangutan", "Gorilla",
    "Chimpanzee", "Baboon", "Marmoset", "Tamarin", "Koala", "Platypus", "Echidna", "Parrot", "Eagle", "Hawk",
    "Falcon", "Owl", "Penguin", "Puffin", "Pelican", "Albatross", "Swan", "Flamingo", "Peacock", "Toucan",
    "Dove", "Pigeon", "Seagull", "Woodpecker", "Hummingbird", "Canary", "Goldfinch", "Crocodile", "Alligator", "Turtle",
    "Tortoise", "Lizard", "Chameleon", "Gecko", "Iguana", "Snake", "Python", "Cobra", "Viper", "Anaconda",
    "Frog", "Toad", "Salamander", "Newt", "Jellyfish", "Octopus", "Squid", "Crab", "Lobster", "Shrimp",
    "Seahorse", "Starfish", "Clownfish", "Shark", "Dolphin", "Whale", "Orca", "Manatee", "Narwhal", "Stingray"
]

def clean_text(text):
    """
    Clean text by removing references and extra spaces.
    """
    text = re.sub(r'\[[0-9]+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_dataset():
    """
    Create a dataset by extracting sentences from Wikipedia articles.
    """
    data = []
    
    for animal in animals:
        page = wiki_wiki.page(animal)
        if page.exists():
            paragraphs = clean_text(page.text).split('. ')
            random.shuffle(paragraphs)
            selected_sentences = [p for p in paragraphs if animal.lower() in p.lower()][:5]
            for sentence in selected_sentences:
                data.append({"text": sentence, "label": animal})
    
    with open(config.NER_DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    df = pd.DataFrame(data)
    print(df.head())
    print(f"Dataset saved successfully as {config.NER_DATASET_JSON}! Total examples: {len(data)}")

if __name__ == "__main__":
    create_dataset()
