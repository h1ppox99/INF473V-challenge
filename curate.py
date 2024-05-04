import hydra
from omegaconf import OmegaConf
import clip
import torch
from PIL import Image
import os
import yaml

@hydra.main(config_path='configs/dataset_curator', config_name='config')
def curate(cfg):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    with open(cfg.cheese_description, 'r') as file:
        cheese_description = yaml.safe_load(file)

    # Derive labels from the names of subdirectories in the specified images directory
    labels = [name for name in os.listdir(cfg.images_dir) if os.path.isdir(os.path.join(cfg.images_dir, name))]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {cheese_description[label]} cheese") for label in labels]).to(device)
    

    
    # Encode the text inputs once, outside of the image loop
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Process images within each subfolder, corresponding to their label
    nb_wrong = 0
    # print(text_inputs[0]==text_inputs[1])
    for label in labels:
        print("Processing label:", label)
        subfolder_path = os.path.join(cfg.images_dir, label)
        image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        ##debug
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)

                # Calculate similarities and determine the best label
                similarities = torch.matmul(text_features, image_features.T).squeeze()
            #     #choose the 10 best predictions
            #     best_label_idx = similarities.argsort(descending=True)[:10]
            # # Log the results
            # print(f"Image {image_file} in folder '{label}' is best described by labels '{[labels[i] for i in best_label_idx]}'.")
            # if not(label in [labels[i] for i in best_label_idx]):
            #     nb_wrong += 1
            k = min(cfg.nb_list, similarities.size(0))
            
            top_similarities, top_indices = torch.topk(similarities, k)

            # Get the labels of the top 10 similarities
            top_labels = [labels[i] for i in top_indices]

            # Log the results
            # print(f"Image {image_file} in folder '{label}' is best described by labels '{top_labels}'.")
            if label not in top_labels:
                nb_wrong += 1
                # print(f"Image {image_file} in folder '{label}' is best described by label '{best_label}'.")
            else: #copy the image to the curated folder /dataset/train/curated/
                # Create the curated folder if it doesn't exist
                curated_folder = os.path.join(cfg.data_dir, 'train', 'curated', label)
                os.makedirs(curated_folder, exist_ok=True)
                #move the image to the curated folder
                os.rename(image_path, os.path.join(curated_folder, image_file))
    print(f"Number of wrong predictions: {nb_wrong}")

@hydra.main(config_path='configs/dataset_curator', config_name='config')
def moveback(cfg):
    # Move the images back to their original folders
    curated_folder = os.path.join(cfg.data_dir, 'train', 'curated')
    labels = [name for name in os.listdir(curated_folder) if os.path.isdir(os.path.join(cfg.images_dir, name))]
    for label in labels:
        curated_folder = os.path.join(cfg.data_dir, 'train', 'curated', label)
        curated_files = os.listdir(curated_folder)
        for curated_file in curated_files:
            os.rename(os.path.join(curated_folder, curated_file), os.path.join(cfg.images_dir, label, curated_file))
        os.rmdir(curated_folder)
                

                

if __name__ == "__main__":
    curate()