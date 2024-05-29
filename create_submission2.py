import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import torch
import re
from fuzzywuzzy import fuzz
import pytesseract
import os
import torch.nn.functional as F

os.environ['TESSDATA_PREFIX'] = '/users/eleves-b/2022/edouard.rabasse/tessdata'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


def preprocess_image(image):
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.point(lambda p: 255 if p > 128 else 0)
    image = image.resize([int(dim * 2) for dim in image.size], Image.LANCZOS)
    image = image.filter(ImageFilter.MedianFilter())
    return image

def clean_text(text):
    words = re.findall(r'\b[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]{3,}\b', text)
    cleaned_text = ' '.join(words)
    return cleaned_text

def extract_text_from_image(image):
    custom_config = r'--oem 3 --psm 11 -l fra'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text if text.strip() else "No text recognised"

def compute_similarity_scores(text, cheese_names):
    scores = {}
    for cheese in cheese_names:
        score = fuzz.partial_ratio(text.lower(), cheese.lower()) / 100  # Normalize score between 0 and 1
        scores[cheese] = score
    best_cheese = max(scores, key=scores.get)
    best_score = scores[best_cheese]
    if best_score > 0.8:
        return best_cheese, best_score
    else:
        return "No cheese matched", 0

cheese_names = [
    "BRIE DE MELUN", "CAMEMBERT", "EPOISSES", "FOURME D’AMBERT", "RACLETTE",
    "MORBIER", "SAINT-NECTAIRE", "POULIGNY SAINT- PIERRE", "ROQUEFORT", "COMTÉ",
    "CHÈVRE", "PECORINO", "NEUFCHATEL", "CHEDDAR", "BÛCHETTE DE CHÈVRE",
    "PARMESAN", "SAINT- FÉLICIEN", "MONT D’OR", "STILTON", "SCARMOZA",
    "CABECOU", "BEAUFORT", "MUNSTER", "CHABICHOU", "TOMME DE VACHE",
    "REBLOCHON", "EMMENTAL", "FETA", "OSSAU- IRATY", "MIMOLETTE",
    "MAROILLES", "GRUYÈRE", "MOTHAIS", "VACHERIN", "MOZZARELLA",
    "TÊTE DE MOINES", "FRAIS"
]


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
    class_names = sorted(os.listdir(cfg.dataset.train_path))

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        output = model(images)
        output = F.softmax(output, dim=1)
        
        certainties, preds = torch.max(output, dim=1)

        preds = [class_names[pred] for pred in preds.cpu().numpy()]
        certainties = certainties.detach().cpu().numpy()

        for j, image_name in enumerate(image_names):
            print("Certainty: ", certainties[j])
            if certainties[j] < 0.8:
                print(f'Using OCR on image number {j}')
                image_path = os.path.join(cfg.dataset.test_path, image_name + ".jpg")
                original_image = Image.open(image_path)
                preprocessed_image = preprocess_image(original_image)
                extracted_text = extract_text_from_image(preprocessed_image)
                cleaned_text = clean_text(extracted_text)
                best_cheese, best_score = compute_similarity_scores(cleaned_text, cheese_names)

                if best_score > 0.8:
                    final_label = best_cheese
                else:
                    final_label = preds[j]
            else:
                final_label = preds[j]


            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"id": [image_name], "label": [final_label]}),
                ]
            )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)


if __name__ == "__main__":
    create_submission()
