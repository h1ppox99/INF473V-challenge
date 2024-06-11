#########################
## MODULES NÉCESSAIRES ##
#########################
import hydra
from omegaconf import OmegaConf
import clip
import torch
from PIL import Image
import os
import yaml
import torch.nn.functional as F



##########################
## FONCTION DE CURATION ##
##########################
    
@hydra.main(config_path='configs/dataset_curator', config_name='config')
def curate2(cfg):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Charge le modèle CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    with open(cfg.cheese_description, 'r') as file:
        cheese_description = yaml.safe_load(file)
    
    

    # Extrait les labels des noms des sous-répertoires dans le répertoire d'images spécifié
    labels = [name for name in os.listdir(cfg.images_dir) if os.path.isdir(os.path.join(cfg.images_dir, name))]
 
    text_inputs = torch.cat([clip.tokenize(f"{label} cheese") for label in labels]).to(device)
    
    # utiliser le jeu de validation pour curer le jeu d'entraînement
    # Traiter les images dans le jeu de validation
    val_input = os.path.join(cfg.data_dir, 'val')
    val_labels = [name for name in os.listdir(val_input) if os.path.isdir(os.path.join(val_input, name))]

    # Encoder les images de validation une fois, en dehors de la boucle d'images
    with torch.no_grad():
        val_image_features = []
        for label in val_labels:
            label_features = []
            subfolder_path = os.path.join(val_input, label)
            image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                image_path = os.path.join(subfolder_path, image_file)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                label_features.append(image_features)
            # Prendre la moyenne des caractéristiques de l'image pour le label
            label_features = torch.stack(label_features).mean(dim=0)
            val_image_features.append(label_features)
        val_image_features = torch.cat(val_image_features)
    

    
    # Encoder le texte
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    # Choisir les caractéristiques à utiliser
    if cfg.use_val:
        features = val_image_features
    else:
        features = text_features

    # Comparer les images dans les sous-répertoires des labels
    nb_wrong = 0
    results = {}
    for label in labels:
        index = labels.index(label)
        print("Processing label:", label)
        nb_wrong_label = 0
        nb_total = 0
        subfolder_path = os.path.join(cfg.images_dir, label)
        image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        best_labels = []
        mean_similarity = 0
        nb_label = 0
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            nb_total += 1
            nb_label += 1

            with torch.no_grad():
                image_features = model.encode_image(image)

                # Calculer la similarité cosinus entre l'image et le label
                similarity = F.cosine_similarity(image_features, features[index].unsqueeze(0)).item()
                # add to result txt
                mean_similarity += similarity

                # Si la similarité est inférieure à un seuil, considérer l'image mal étiquetée
                threshold = cfg.threshold  # seuil de similarité
                if similarity < threshold:
                    if cfg.delete:
                        # Supprimer l'image
                        os.remove(image_path)
                    nb_wrong_label += 1
                    nb_wrong += 1
        print(f"Mean similarity for {label}: {mean_similarity/nb_total}")
                
        print(f"Number of wrong predictions for {label}: {nb_wrong_label} out of {nb_total}")
        results[label] = f"Number of wrong predictions for {label}: {nb_wrong_label} out of {nb_total} \n"
    # Sauvegarder les résultats
    with open("results.txt", "w") as file:
        file.write(str(results))

if __name__ == "__main__":
    curate2()