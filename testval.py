import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValDataset(Dataset):
    def __init__(self, val_dataset_path, val_transform):
        self.val_dataset_path = val_dataset_path
        self.val_transform = val_transform
        self.images_list = []
        self.labels_list = []
        for label in os.listdir(self.val_dataset_path):
            label_path = os.path.join(self.val_dataset_path, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.images_list.append(os.path.join(label_path, image_name))
                        self.labels_list.append(label)

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        label = self.labels_list[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.val_transform(image)
        return image, label

    def __len__(self):
        return len(self.images_list)


def adjust_scores(preds, class_names, target_mean=0.027):
    """
    Adjusts the scores in the preds tensor to normalize their means to the target_mean.
    
    Args:
    - preds (torch.Tensor): The tensor of predicted scores.
    - class_names (list): The list of class names corresponding to the columns in preds.
    - target_mean (float): The target mean value to adjust the scores to (default is 0.027).
    
    Returns:
    - adjusted_preds (torch.Tensor): The tensor with adjusted scores.
    """
    # Average scores from your data
    average_scores = {
        "BEAUFORT": 0.0337,
        "BRIE DE MELUN": 0.0299,
        "BÛCHETTE DE CHÈVRE": 0.0178,
        "CABECOU": 0.0265,
        "CAMEMBERT": 0.0166,
        "CHABICHOU": 0.0219,
        "CHEDDAR": 0.0358,
        "CHÈVRE": 0.0223,
        "COMTÉ": 0.0197,
        "EMMENTAL": 0.0220,
        "EPOISSES": 0.0260,
        "FETA": 0.0285,
        "FOURME D’AMBERT": 0.0277,
        "FROMAGE FRAIS": 0.0314,
        "GRUYÈRE": 0.0410,
        "MAROILLES": 0.0192,
        "MIMOLETTE": 0.0289,
        "MONT D’OR": 0.0353,
        "MORBIER": 0.0306,
        "MOTHAIS": 0.0255,
        "MOZZARELLA": 0.0224,
        "MUNSTER": 0.0200,
        "NEUFCHATEL": 0.0261,
        "OSSAU- IRATY": 0.0265,
        "PARMESAN": 0.0313,
        "PECORINO": 0.0276,
        "POULIGNY SAINT- PIERRE": 0.0390,
        "RACLETTE": 0.0369,
        "REBLOCHON": 0.0201,
        "ROQUEFORT": 0.0322,
        "SAINT- FÉLICIEN": 0.0223,
        "SAINT-NECTAIRE": 0.0283,
        "SCARMOZA": 0.0216,
        "STILTON": 0.0276,
        "TOMME DE VACHE": 0.0318,
        "TÊTE DE MOINES": 0.0292,
        "VACHERIN": 0.0166
    }
    
    # Calculate the scaling factors
    scaling_factors = torch.tensor([target_mean / average_scores[class_name] for class_name in class_names], dtype=torch.float32, device=preds.device)
    
    # Adjust the preds tensor
    adjusted_preds = preds * scaling_factors
    
    return adjusted_preds


@hydra.main(config_path="configs/train", config_name="config")
def validate_model(cfg):
    val_loader = DataLoader(
        ValDataset(
            cfg.datamodule.real_images_val_path, hydra.utils.instantiate(cfg.datamodule.val_transform)
        ),
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
    )

    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    
    # Filter out non-directory entries
    class_names = sorted([d for d in os.listdir(cfg.datamodule.train_dataset_path) if os.path.isdir(os.path.join(cfg.datamodule.train_dataset_path, d))])
    print(f"Filtered class names: {class_names}")

    correct_predictions = 0
    total_predictions = 0

    for i, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device)
        preds = model(images)
        softmax_preds = torch.nn.functional.softmax(preds, dim=1)
        softmax_preds = adjust_scores(softmax_preds, class_names)
        
        max_preds = softmax_preds.argmax(1)
        predicted_labels = [class_names[pred] for pred in max_preds.cpu().numpy()]

        correct_predictions += sum([pred == true_label for pred, true_label in zip(predicted_labels, labels)])
        total_predictions += len(labels)

    accuracy = correct_predictions / total_predictions
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    validate_model()
