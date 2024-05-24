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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)

text_recognition = TextRecognition('/path/to/tessdata_fast', list(cheese_keywords.keys()), cheese_keywords)


@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg: DictConfig):
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
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    
    # Inspect the keys of the checkpoint and adjust accordingly
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    class_names = sorted(os.listdir(cfg.dataset.train_path))

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        # Indiquer le nombre de batchs restants
        print(f"Batch {i}/{len(test_loader)}")
        images, image_names = batch
        images = images.to(device)
        with torch.no_grad():
            preds = model(images)
        preds = preds.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]

        for j, image_name in enumerate(image_names):
            # Indiquer le nombre d'images restantes
            print(f"Image {j}/{len(image_names)}")
            image_path = os.path.join(cfg.dataset.test_path, image_name + ".jpg")
            original_image = Image.open(image_path)
            best_cheese, best_score = text_recognition.predict(image_path, preprocess)

            if best_score > 0.7:
                final_label = best_cheese
            else:
                final_label = preds[j]

            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"id": [image_name], "label": [final_label]}),
                ],
                ignore_index=True
            )
    submission.to_csv(os.path.join(cfg.root_dir, "submission.csv"), index=False)

if __name__ == "__main__":
    create_submission()
