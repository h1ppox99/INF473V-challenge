import os
import re
import easyocr
from PIL import Image
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from PIL import ImageEnhance

def preprocess(image, resize_factor=1.5):
    image = image.resize([int(dim * resize_factor) for dim in image.size], Image.LANCZOS)
    return image


class TextRecognition:
    def __init__(self, tessdata_prefix, cheese_names, cheese_keywords):
        self.tessdata_prefix = tessdata_prefix
        os.environ['TESSDATA_PREFIX'] = tessdata_prefix
        self.cheese_names = cheese_names
        self.cheese_keywords = cheese_keywords
        self.reader = easyocr.Reader(['fr', 'en'], gpu=torch.cuda.is_available())

    def clean_text(self, text):
        words = re.findall(r'\b[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]{3,}\b', text)
        cleaned_text = ' '.join(words)
        return cleaned_text

    def extract_text_from_image(self, preprocessed_image):
        result = self.reader.readtext(np.array(preprocessed_image))
        combined_text = ' '.join([item[1] for item in result])
        return combined_text if combined_text.strip() else ""

    def predict(self, image_path, preprocess_function):
        with Image.open(image_path) as img:
            img = img.convert('L')
            img = preprocess_function(img)
            extracted_text = self.extract_text_from_image(img)
            cleaned_text = self.clean_text(extracted_text)
            best_cheese, best_score = self.compute_similarity_scores(cleaned_text)
            return best_cheese, best_score

    def compute_similarity_scores(self, text):
        scores = {}
        for cheese in self.cheese_names:
            keywords = self.cheese_keywords[cheese]
            max_score = max(fuzz.partial_ratio(text.lower(), keyword.lower()) / 100 for keyword in keywords)
            scores[cheese] = max_score
        best_cheese = max(scores, key=scores.get)
        best_score = scores[best_cheese]
        return best_cheese, best_score

# Liste des fromages avec leurs mots-clés
cheese_keywords = {
    "BRIE DE MELUN": ["brie", "melun","seine-et-marne"],
    "CAMEMBERT": ["camembert", "normandie", "calvados","president"],
    "EPOISSES": ["époisses", "burgundy", "affiné","berthaut"],
    "FOURME D’AMBERT": ["fourme", "ambert"],
    "RACLETTE": ["raclette"],
    "MORBIER": ["morbier"],
    "SAINT-NECTAIRE": ["saint-nectaire", "auvergne", "cantal"],
    "POULIGNY SAINT- PIERRE": ["pouligny", "saint-pierre", "pyramide"],
    "ROQUEFORT": ["roquefort"],
    "COMTÉ": ["comté"],
    "CHÈVRE": ["zzzzzzzzzzzzzzzzzzzzzzzzzzzz"], # On ne met pas de mots-clés pour le fromage de chèvre car cela fait des faux positifs
    "PECORINO": ["pecorino", "romano"],
    "NEUFCHATEL": ["neufchâtel", "brais"],
    "CHEDDAR": ["cheddar", "mature"],
    "BÛCHETTE DE CHÈVRE": ["zzzzzzzzzzzzzzzzzzzzzzzz"], # On ne met pas de mots-clés pour le fromage de chèvre car cela fait des faux positifs
    "PARMESAN": ["parmesan", "parmigiano", "reggiano"],
    "SAINT-FÉLICIEN": ["saint-felicien", "felicien"],
    "MONT D’OR": ["mont d’or", "haut-doubs", "arnaud"],
    "STILTON": ["stilton", "blue", "england"],
    "SCARMOZA": ["scarmoza", "affumicata"],
    "CABECOU": ["cabecou", "rocamadour"],
    "BEAUFORT": ["beaufort"],
    "MUNSTER": ["munster", "alsace", "cumin"],
    "CHABICHOU": ["chabichou", "poitou"],
    "TOMME DE VACHE": ["tomme", "cow", "montagne"],
    "REBLOCHON": ["reblochon", "savoie"],
    "EMMENTAL": ["emmental"],
    "FETA": ["feta", "greek", "grecque"],
    "OSSAU- IRATY": ["ossau", "iraty"],
    "MIMOLETTE": ["mimolette", "vieille"],
    "MAROILLES": ["maroilles"],
    "GRUYÈRE": ["gruyère", "switzerland"],
    "MOTHAIS": ["mothais", "sur feuille"],
    "VACHERIN": ["vacherin"],
    "MOZZARELLA": ["mozzarella", "di bufala", "campana"],
    "TÊTE DE MOINES": ["tête de moine", "moine", "bellelay"],
    "FROMAGE FRAIS": ["fromage frais", "fresh cheese", "nature"]
}

'''
text_recognition = TextRecognition('/path/to/tessdata_fast', list(cheese_keywords.keys()), cheese_keywords)

val_dir = "/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/dataset/val/"

preprocess_function = preprocess

# Collect results for all images
results = []

for cheese_type in os.listdir(val_dir):
    cheese_dir = os.path.join(val_dir, cheese_type)
    for image_name in os.listdir(cheese_dir):
        image_path = os.path.join(cheese_dir, image_name)
        prediction, best_score = text_recognition.predict(image_path, preprocess_function)
        results.append((cheese_type, prediction, best_score))

# Apply thresholds and calculate percentage of answered images
thresholds = np.arange(0.4, 1.0, 0.01)
accuracies = []

for threshold in thresholds:
    true_labels = []
    predicted_labels = []
    
    for true_label, predicted_label, score in results:
        if score > threshold:
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
    
    if true_labels:
        accuracy = accuracy_score(true_labels, predicted_labels)
    else:
        accuracy = 0
    accuracies.append(accuracy)



# Plot accuracy vs threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, marker='o')
plt.title('Accuracy vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/accuracy_vs_threshold.png")

'''
