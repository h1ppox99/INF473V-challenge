import os
import re
import easyocr
from PIL import Image
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
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
        words = re.findall(r'\b[a-zA-ZàäëïîôöùûüÿçÀÂÄÊËÏÎÔÖÙÛÜŸÇ]{3,}\b', text)
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
            cheese_scores = [fuzz.partial_ratio(text.lower(), keyword.lower()) / 100 for keyword in keywords]
            scores[cheese] = max(cheese_scores) if cheese_scores else 0

        best_cheese = max(scores, key=scores.get)
        best_score = scores[best_cheese]
        return best_cheese, best_score





# Liste des fromages avec leurs mots-clés
cheese_keywords = {
    "BRIE DE MELUN": ["brie", "melun","seine-et-marne"],
    "CAMEMBERT": ["camembert", "normandie", "calvados","president"],
    "EPOISSES": ["époisses", "burgundy", "berthaut"],
    "FOURME D’AMBERT": ["fourme", "ambert"],
    "RACLETTE": ["raclette"],
    "MORBIER": ["morbier"],
    "SAINT-NECTAIRE": ["saint-nectaire"],
    "POULIGNY SAINT- PIERRE": ["pouligny", "saint-pierre", "pyramide"],
    "ROQUEFORT": ["roquefort","société"],
    "COMTÉ": ["comté"],
    "CHÈVRE": ["zzzzzzzzzzzzzzzzzzzzzzzzz"], 
    "PECORINO": ["pecorino", "romano"],
    "NEUFCHATEL": ["neufchâtel", "brais"],
    "CHEDDAR": ["cheddar", "mature"],
    "BÛCHETTE DE CHÈVRE": ["zzzzzzzzzzzzzzzzzzz"], # On ne met pas de mots-clés pour le fromage de chèvre car cela fait des faux positifs
    "PARMESAN": ["parmesan", "parmigiano", "reggiano"],
    "SAINT- FÉLICIEN": ["saint-félicien", "félicien"],
    "MONT D’OR": ["mont d’or", "haut-doubs", "arnaud"],
    "STILTON": ["stilton", "blue", "england"],
    "SCARMOZA": ["scarmoza", "affumicata"],
    "CABECOU": ["cabecou", "rocamadour"],
    "BEAUFORT": ["beaufort"],
    "MUNSTER": ["munster", "alsace", "cumin"],
    "CHABICHOU": ["chabichou", "poitou"],
    "TOMME DE VACHE": ["tomme", "tomme de montagne"],
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

text_recognition = TextRecognition('/path/to/tessdata_fast', list(cheese_keywords.keys()), cheese_keywords)

val_dir = "/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/dataset/val/"

preprocess_function = preprocess
'''
# Collect results for all images
results = []

for cheese_type in os.listdir(val_dir):
    cheese_dir = os.path.join(val_dir, cheese_type)
    for image_name in os.listdir(cheese_dir):
        image_path = os.path.join(cheese_dir, image_name)
        prediction, best_score = text_recognition.predict(image_path, preprocess_function)
        results.append((cheese_type, prediction, best_score))

thresholds = np.arange(0.6, 0.91, 0.01)
accuracies = []
response_rates = []
products = []

for threshold in thresholds:
    true_labels = []
    predicted_labels = []

    for true_label, predicted_label, score in results:
        if score > threshold and score < threshold + 0.05:
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    response_rate = len(predicted_labels) / len(results)
    accuracies.append(accuracy)
    response_rates.append(response_rate)
    

# Plotting accuracy, response rate, and product of both
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(thresholds, accuracies, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Response Rate', color=color)
ax2.plot(thresholds, response_rates, color=color, label='Response Rate', linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()
plt.title('Accuracy and Response Rate vs Threshold')
fig.legend(loc='upper left')
plt.savefig("test_zeroclean.png")
plt.show()
'''