#########################
## MODULES NÉCESSAIRES ##
#########################

import os
import re
import pandas as pd
import torch
import sys
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from fuzzywuzzy import fuzz
import pytesseract
import hydra
from omegaconf import DictConfig
from text_recognition_easyocr import *
import time

sys.path.append('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/ocr')
from text_recognition_easyocr import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################
## FONCTION AUXILIAIRE ##
#########################

# Scores moyens 
# À MODIFIER A POSTERIORI APRÈS CHAQUE NOUVELLE SOUMISSION

average_scores = {
    "BEAUFORT": 0.0472,
    "BRIE DE MELUN": 0.0269,
    "BÛCHETTE DE CHÈVRE": 0.0176,
    "CABECOU": 0.0247,
    "CAMEMBERT": 0.0174,
    "CHABICHOU": 0.0297,
    "CHEDDAR": 0.0332,
    "CHÈVRE": 0.0176,
    "COMTÉ": 0.0216,
    "EMMENTAL": 0.0282,
    "EPOISSES": 0.0205,
    "FETA": 0.0331,
    "FOURME D’AMBERT": 0.0311,
    "FROMAGE FRAIS": 0.0234,
    "GRUYÈRE": 0.0266,
    "MAROILLES": 0.0261,
    "MIMOLETTE": 0.0338,
    "MONT D’OR": 0.0403,
    "MORBIER": 0.0346,
    "MOTHAIS": 0.0287,
    "MOZZARELLA": 0.0275,
    "MUNSTER": 0.0213,
    "NEUFCHATEL": 0.0265,
    "OSSAU- IRATY": 0.0200,
    "PARMESAN": 0.0312,
    "PECORINO": 0.0178,
    "POULIGNY SAINT- PIERRE": 0.0371,
    "RACLETTE": 0.0388,
    "REBLOCHON": 0.0197,
    "ROQUEFORT": 0.0301,
    "SAINT- FÉLICIEN": 0.0224,
    "SAINT-NECTAIRE": 0.0272,
    "SCARMOZA": 0.0222,
    "STILTON": 0.0302,
    "TOMME DE VACHE": 0.0275,
    "TÊTE DE MOINES": 0.0246,
    "VACHERIN": 0.0135
}

def adjust_scores(preds, class_names, target_mean=0.027):
    """
    Ajuste les scores dans le tenseur preds pour normaliser leurs moyennes à target_mean.
    
    Args:
    - preds (torch.Tensor): Le tenseur des scores prédits.
    - class_names (liste): La liste des noms de classes correspondant aux colonnes dans preds.
    - target_mean (float): La valeur moyenne cible pour ajuster les scores (par défaut 0.027).
    
    Retourne:
    - adjusted_preds (torch.Tensor): Le tenseur avec les scores ajustés.
    """
    # Calcul des facteurs de mise à l'échelle
    scaling_factors = torch.tensor([target_mean / average_scores[class_name] for class_name in class_names], dtype=torch.float32, device=preds.device)
    
    # Ajustement du tenseur preds
    adjusted_preds = preds * scaling_factors
    
    return adjusted_preds


#####################
## CLASSE DATASET ##
#####################


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # Filtrer les fichiers non-images
        # Problèmes rencontrés avec les fichiers .DS_Store sur macOS
        self.images_list = [image for image in images_list si image.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)

@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg):
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )

    # Charger le modèle et le checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Chargement du modèle depuis le checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    
    # Filtrer les entrées non-répertoires
    class_names = sorted([d for d in os.listdir(cfg.dataset.train_path) si os.path.isdir(os.path.join(cfg.dataset.train_path, d))])
    print(f"Noms des classes filtrées: {class_names}")

    # Créer un DataFrame pour la soumission
    submission = pd.DataFrame(columns=["id", "label"])

    df1 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submissions/submission_ocr83_2.csv')
    df1.set_index("id", inplace=True)  # Définir l'index sur 'id' pour une recherche rapide
    
    # Dictionnaire pour suivre les scores totaux et les comptes pour chaque classe
    model_class_scores = {class_name: 0.0 for class_name in class_names}
    model_class_counts = {class_name: 0 for class_name in class_names}

    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        preds = model(images)
        softmax_preds = torch.nn.functional.softmax(preds, dim=1)
        softmax_preds = adjust_scores(softmax_preds, class_names)
        
        for j in range(softmax_preds.size(0)):
            for k, class_name in enumerate(class_names):
                image_id = image_names[j]
                csv_label = df1.loc[image_id, "label"]
                if csv_label == "UNKNOWN":
                    model_class_scores[class_name] += softmax_preds[j, k].item()
                    model_class_counts[class_name] += 1
        
        max_preds = softmax_preds.argmax(1)

        labels = []
        '''
        for j, image_name in enumerate(image_names):
            # Indiquer le nombre d'images restantes
            print(f"Image {j}/{len(image_names)}")
            # On récupère le score de la prédiction
            #model_score = scores[j][class_names.index(preds[j])].item()
            image_path = os.path.join(cfg.dataset.test_path, image_name + ".jpg")
            original_image = Image.open(image_path)
            best_cheese, best_score = text_recognition.predict(image_path, preprocess)

            if (best_score > 0.83):
                labels.append(best_cheese)
            else:
                labels.append("UNKNOWN")
        '''  
        
        labels = []
        for idx, pred in enumerate(max_preds):
            image_id = image_names[idx]
            csv_label = df1.loc[image_id, "label"]
            if csv_label != "UNKNOWN":
                labels.append(csv_label)
            else:
                labels.append(class_names[pred])
        
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": labels}),
            ],
            ignore_index=True  # S'assurer que les indices sont continus
        )

    # Calculer le score moyen pour chaque classe en utilisant uniquement les prédictions du modèle
    model_class_averages = {class_name: (model_class_scores[class_name] / model_class_counts[class_name]) si model_class_counts[class_name] > 0 else 0 for class_name in class_names}
    
    # Imprimer les scores moyens pour les prédictions du modèle
    for class_name, average_score in model_class_averages.items():
        print(f"Score moyen pour {class_name} (prédiction du modèle): {average_score:.4f}")

    # Trier le DataFrame de soumission par 'id' pour garantir une sortie triée
    submission = submission.sort_values(by='id').reset_index(drop=True)

    # Sauvegarder le DataFrame de soumission trié dans un fichier CSV
    submission.to_csv(f"{cfg.root_dir}/submission_30_05_05.csv", index=False)
    print(f"Soumission sauvegardée dans {cfg.root_dir}/submission_30_05_05.csv")

if __name__ == "__main__":
    create_submission()
