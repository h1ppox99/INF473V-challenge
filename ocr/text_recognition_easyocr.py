#########################
## MODULES NÉCESSAIRES ##
#########################

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
import time


##########################
## FONCTIONS AUXILAIRES ##
##########################


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
        words = re.findall(r'\b[a-zA-ZàäëïîôöùüÿçÀÂÄÊËÏÎÔÖÙÜŸÇ]{3,}\b', text)
        cleaned_text = ' '.join(words)
        print(cleaned_text)
        return cleaned_text

    def extract_text_from_image(self, preprocessed_image):
        result = self.reader.readtext(np.array(preprocessed_image))
        combined_text = ' '.join([item[1] for item in result])
        print(combined_text)
        return combined_text if combined_text.strip() else ""

    '''
    Méthode abandonnée : l'idée était de ne pas comparer le texte brut mais de comparer les n-grams des
    mots-clés des fromages avec les n-grams du texte extrait de l'image
    

    def generate_ngrams(self, text, n):
        words = text.split()
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        return ngrams
    '''

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
        #print("Score :"+str(best_score)+" Fromage :"+best_cheese)
        return best_cheese, best_score



# Liste des fromages avec leurs mots-clés
cheese_keywords = {
    "BRIE DE MELUN": ["Brie", "Melun", "Brie de Melun", "Seine-et-Marne", "BRIE", "MELUN", "SEINE-ET-MARNE", "BRIE DE MELUN"],
    "CAMEMBERT": ["camembert", "Normandie", "Calvados", "CAMEMBERT", "NORMANDIE", "CALVADOS"],
    "EPOISSES": ["Époisses", "Epoisse", "Burgundy", "Berthaut", "ÉPOISSES", "BURGUNDY", "BERTHAUT", "EPOISSES"],
    "FOURME D’AMBERT": ["Fourme d'Ambert","Ambert", "AMBERT", "FOURME D’AMBERT"],
    "RACLETTE": ["Raclette", "RACLETTE"],
    "MORBIER": ["Morbier", "MORBIER"],
    "SAINT-NECTAIRE": ["Saint-Nectaire", "SAINT-NECTAIRE"],
    "POULIGNY SAINT- PIERRE": ["Pouligny Saint-Pierre","Pouligny", "Saint-Pierre", "pyramide", "POULIGNY", "SAINT-PIERRE", "PYRAMIDE", "POULIGNY SAINT-PIERRE"],
    "ROQUEFORT": ["roquefort", "société", "ROQUEFORT", "SOCIÉTÉ"],
    "COMTÉ": ["comte", "COMTE", "comté", "COMTÉ"],
    "CHÈVRE": ["zzzzzzzzzzzzzzzzzzzzzzzzz"],
    "PECORINO": ["Pecorino", "romano", "Pecorino romano", "PECORINO", "ROMANO", "PECORINO ROMANO"],
    "NEUFCHATEL": ["Neufchâtel", "Pays de Brais", "NEUFCHÂTEL", "PAYS DE BRAIS"],
    "CHEDDAR": ["Cheddar", "Cheddar mature", "CHEDDAR", "CHEDDAR MATURE"],
    "BÛCHETTE DE CHÈVRE": ["Bûche", "Bûchette", "Soignon", "BÛCHE", "BÛCHETTE", "SOIGNON"],
    "PARMESAN": ["Parmesan", "Parmigiano", "reggiano", "Pamiggiano reggiano", "PARMESAN", "PARMIGIANO", "REGGIANO", "PAMIGGIANO REGGIANO"],
    "SAINT- FÉLICIEN": ["Saint-Félicien", "St-Félicien", "Félicien", "Étoile du Vercors", "SAINT-FÉLICIEN", "FÉLICIEN", "ÉTOILE DU VERCORS"],
    "MONT D’OR": ["Mont d’or", "Mont", "Haut-Doubs", "Arnaud", "MONT D’OR", "HAUT-DOUBS", "ARNAUD", "MONT D’OR"],
    "STILTON": ["stilton", "blue", "Blue Stilton", "england", "STILTON", "BLUE", "ENGLAND"],
    "SCARMOZA": ["Scarmoza", "affumicata", "Scarmoza affumicata", "SCARMOZA", "AFFUMICATA", "SCARMOZA AFFUMICATA"],
    "CABECOU": ["Cabecou", "Rocamadour", "CABECOU", "ROCAMADOUR"],
    "BEAUFORT": ["Beaufort", "BEAUFORT"],
    "MUNSTER": ["Munster", "Alsace", "MUNSTER", "ALSACE"],
    "CHABICHOU": ["Chabichou", "Poitou", "CHABICHOU", "POITOU"],
    "TOMME DE VACHE": ["Tomme", "Tomme de montagne", "TOMME", "TOMME DE MONTAGNE"],
    "REBLOCHON": ["Reblochon", "Savoie", "REBLOCHON", "SAVOIE"],
    "EMMENTAL": ["Emmental", "EMMENTAL"],
    "FETA": ["feta", "greek", "grecque", "FETA", "GREEK", "GRECQUE"],
    "OSSAU- IRATY": ["Ossau", "iraty", "Ossau-Iraty","OSSAU", "IRATY", "OSSAU-IRATY"],
    "MIMOLETTE": ["Mimolette", "vieille", "MIMOLETTE", "VIEILLE"],
    "MAROILLES": ["Maroilles", "MAROILLES"],
    "GRUYÈRE": ["Gruyère", "Switzerland", "GRUYÈRE", "SWITZERLAND"],
    "MOTHAIS": ["Mothais", "Mothais sur feuille" ,"sur feuille", "MOTHAIS", "SUR FEUILLE"],
    "VACHERIN": ["Vacherin", "fribourgeois", "VACHERIN", "FRIBOURGEOIS"],
    "MOZZARELLA": ["Mozzarella", "di bufala", "Mozarella di buffala", "MOZZARELLA", "DI BUFALA", "CAMPANA"],
    "TÊTE DE MOINES": ["Tête de moine", "bellelay", "TÊTE DE MOINE", "MOINE", "BELLELAY"],
    "FROMAGE FRAIS": ["Fromage frais", "fresh cheese", "Fromage frais nature", "yoplait", "FROMAGE FRAIS", "FRESH CHEESE", "NATURE", "YOPLAIT"]
}

text_recognition = TextRecognition('/path/to/tessdata_fast', list(cheese_keywords.keys()), cheese_keywords)

val_dir = "/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/dataset/val/"

preprocess_function = preprocess


# On reconnait une image
path_image = "/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/dataset/val/CAMEMBERT/000017.jpg"

prediction, best_score = text_recognition.predict(path_image, preprocess_function)
print(prediction, best_score)


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