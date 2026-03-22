import re

animals = ["cow", "dog", "cat", "horse", "sheep", 
           "elephant", "butterfly", "chicken", "spider", "squirrel"]

templates = [
    "There is a {animal} in the picture",
    "I can see a {animal} here",
    "This is a photo of a {animal}",
    "The {animal} is in the image",
    "Look at the {animal} in this picture",
    "A {animal} is present in the photo",
    "The {animal} is visible in the picture",
    "I found a {animal} in this photo",
    "This image contains a {animal}",
    "I saw an {animal} today",
    "The photo clearly shows a {animal} in its natural habitat",
    "There appears to be a {animal} in the photo",
    "This picture features a {animal}",
    "A {animal} can be observed here",
    "The image captures a {animal}",
    "A {animal} is shown in this picture",
    "In this photo, there is a {animal}",
    "The {animal} stands out in the image",
    "Here we have a {animal} in the frame",
    "A {animal} is clearly visible here",
    "The picture includes a {animal}",
    "This scene shows a {animal}",
    "A {animal} appears in the image",
    "The {animal} can be seen in the photo",
    "There’s a {animal} captured in this shot",
    "This photograph depicts a {animal}",
    "A {animal} is noticeable in this image",
    "The image highlights a {animal}",
    "One can notice a {animal} here",
]

def generate_ner_dataset(animals, templates):
    dataset = []
    for animal in animals:
        for template in templates:
            sentence = template.format(animal=animal)
            # токенізуємо речення
            tokens = re.findall(r'\w+', sentence)
            # створюємо мітки
            labels = []
            for token in tokens:
                if token == animal:
                    labels.append("B-ANIMAL")
                else:
                    labels.append("O")
            dataset.append({"tokens": tokens, "labels": labels})
    return dataset

dataset = generate_ner_dataset(animals, templates)
print(f"Total samples: {len(dataset)}")  # скільки?
print(dataset[0])  # перший приклад