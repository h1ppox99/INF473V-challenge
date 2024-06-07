#########################
## MODULES NÉCESSAIRES ##
#########################

import torch
import copy
import hydra
from omegaconf import OmegaConf

#########################
## FONCTION PRINCIPALE ##
#########################

'''
But : merge les poids de deux modèles entraînés pour créer un nouveau modèle
Résultats pas satisfaisants pour des modèles entraînés sur des trainsets différents
Non utilisé au final (sauf pour des submissions infructueuses)
'''

# Définir la fonction pour fusionner les poids
def merge_weights(weight1, weight2, alpha=0.5):
    # Paramètre alpha à modifier pour changer le poids des deux modèles
    return alpha * weight1 + (1 - alpha) * weight2

# Initialiser Hydra
@hydra.main(config_path="configs/train", config_name="config")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger le premier modèle
    model1 = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint1 = torch.load(cfg.checkpoint_path_1, map_location=device)
    print(f"Chargement du modèle depuis le checkpoint: {cfg.checkpoint_path_1}")

    if 'model_state_dict' in checkpoint1:
        model1.load_state_dict(checkpoint1['model_state_dict'])
    else:
        model1.load_state_dict(checkpoint1)
    model1.eval()

    # Charger le deuxième modèle
    model2 = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint2 = torch.load(cfg.checkpoint_path_2, map_location=device)
    print(f"Chargement du modèle depuis le checkpoint: {cfg.checkpoint_path_2}")

    if 'model_state_dict' in checkpoint2:
        model2.load_state_dict(checkpoint2['model_state_dict'])
    else:
        model2.load_state_dict(checkpoint2)
    model2.eval()

    # S'assurer que les deux modèles ont la même architecture
    assert model1.state_dict().keys() == model2.state_dict().keys(), "Les modèles n'ont pas la même architecture"

    # Créer une nouvelle instance de modèle (même architecture que model1 et model2)
    new_model = hydra.utils.instantiate(cfg.model.instance).to(device)

    # Fusionner les poids
    new_state_dict = copy.deepcopy(model1.state_dict())
    for param_tensor in model1.state_dict():
        new_state_dict[param_tensor].copy_(
            merge_weights(model1.state_dict()[param_tensor], model2.state_dict()[param_tensor])
        )

    # Charger les poids fusionnés dans le nouveau modèle
    new_model.load_state_dict(new_state_dict)
    new_model.eval()

    # Sauvegarder le nouveau modèle
    new_model_path = '/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/checkpoints/merged_model_80.pt'  # Spécifier le chemin pour sauvegarder le nouveau modèle
    torch.save(new_model.state_dict(), new_model_path)

    print(f"Nouveau modèle avec poids fusionnés sauvegardé à: {new_model_path}")

if __name__ == "__main__":
    main()
