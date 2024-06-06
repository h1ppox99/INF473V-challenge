import os
import re
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from fuzzywuzzy import fuzz
import pytesseract
import hydra
from omegaconf import DictConfig
from text_recognition_easyocr import *
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Average scores from your data
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
    Adjusts the scores in the preds tensor to normalize their means to the target_mean.
    
    Args:
    - preds (torch.Tensor): The tensor of predicted scores.
    - class_names (list): The list of class names corresponding to the columns in preds.
    - target_mean (float): The target mean value to adjust the scores to (default is 0.027).
    
    Returns:
    - adjusted_preds (torch.Tensor): The tensor with adjusted scores.
    """
    # Calculate the scaling factors
    scaling_factors = torch.tensor([target_mean / average_scores[class_name] for class_name in class_names], dtype=torch.float32, device=preds.device)
    
    # Adjust the preds tensor
    adjusted_preds = preds * scaling_factors
    
    return adjusted_preds



class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # Filter out non-image files
        self.images_list = [image for image in images_list if image.lower().endswith((".jpg", ".jpeg", ".png"))]

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

    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    
    # Filter out non-directory entries
    class_names = sorted([d for d in os.listdir(cfg.dataset.train_path) if os.path.isdir(os.path.join(cfg.dataset.train_path, d))])
    print(f"Filtered class names: {class_names}")

    # Create submission DataFrame
    submission = pd.DataFrame(columns=["id", "label"])

    df1 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submissions/submission_ocr83_2.csv')
    df1.set_index("id", inplace=True)  # Set index to 'id' for quick lookup
    
    # Dictionary to track the total scores and counts for each class
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
            ignore_index=True  # Ensure indices are continuous
        )

    # Compute the average score for each class using model predictions only
    model_class_averages = {class_name: (model_class_scores[class_name] / model_class_counts[class_name]) if model_class_counts[class_name] > 0 else 0 for class_name in class_names}
    
    # Print the average scores for model predictions
    for class_name, average_score in model_class_averages.items():
        print(f"Average score for {class_name} (model prediction): {average_score:.4f}")

    # Sort the submission DataFrame by 'id' to ensure sorted output
    submission = submission.sort_values(by='id').reset_index(drop=True)

    # Save the sorted submission DataFrame to a CSV file
    submission.to_csv(f"{cfg.root_dir}/submission_30_05_05.csv", index=False)
    print(f"Submission saved to {cfg.root_dir}/submission_30_05_05.csv")


if __name__ == "__main__":
    create_submission()
