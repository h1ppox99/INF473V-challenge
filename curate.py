import hydra
from omegaconf import OmegaConf
import clip
import torch
from PIL import Image
import os

@hydra.main(config_path='configs/dataset_curator', config_name='config')
def curate(cfg):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # Derive labels from the names of subdirectories in the specified images directory
    labels = [name for name in os.listdir(cfg.images_dir) if os.path.isdir(os.path.join(cfg.images_dir, name))]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {label} cheese") for label in labels]).to(device)

    # Process images within each subfolder, corresponding to their label
    nb_wrong = 0
    for label in labels:
        subfolder_path = os.path.join(cfg.images_dir, label)
        image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_inputs)

                # Calculate similarities and determine the best label
                similarities = torch.matmul(text_features, image_features.T).squeeze(0)
                #choose the 10 best predictions
                best_label_idx = similarities.argsort(descending=True)[:10]
            # Log the results
            if not(label in [labels[i] for i in best_label_idx]):
                nb_wrong += 1
                # print(f"Image {image_file} in folder '{label}' is best described by label '{best_label}'.")
            else: #copy the image to the curated folder /dataset/train/curated/
                # Create the curated folder if it doesn't exist
                curated_folder = os.path.join(cfg.data_dir, 'train', 'curated', label)
                os.makedirs(curated_folder, exist_ok=True)
                os.rename(image_path, os.path.join(curated_folder, image_file))
                #copy the file to the curated folder
    print(f"Number of wrong predictions: {nb_wrong}")
                

                

if __name__ == "__main__":
    curate()