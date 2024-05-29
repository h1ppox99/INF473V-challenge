import hydra
from omegaconf import OmegaConf
import clip
import torch
from PIL import Image
import os
import yaml
import torch.nn.functional as F

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
    
    # Use the validation set to curate the training set
    # Process the images in the validation set
    val_input = os.path.join(cfg.data_dir, 'val')
    val_labels = [name for name in os.listdir(val_input) if os.path.isdir(os.path.join(val_input, name))]

    # Encode the validation images once, outside of the image loop
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
            # Average the image features for the label
            label_features = torch.stack(label_features).mean(dim=0)
            val_image_features.append(label_features)
        val_image_features = torch.cat(val_image_features)
    

    
    # Encode the text inputs once, outside of the image loop
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    # Choose text or val_image_features
    if cfg.use_val:
        features = val_image_features
    else:
        features = text_features

    # Process images within each subfolder, corresponding to their label
    nb_wrong = 0
    # print(text_inputs[0]==text_inputs[1])
    for label in labels:
        print("Processing label:", label)
        nb_wrong_label = 0
        subfolder_path = os.path.join(cfg.images_dir, label)
        image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        best_labels = []
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)

                # Calculate similarities and determine the best label
                similarities = torch.matmul(features, image_features.T).squeeze()

            k = min(cfg.nb_list, similarities.size(0))
            
            top_similarities, top_indices = torch.topk(similarities, k)
      

            # Get the labels of the top 10 similarities
            top_labels = [labels[i] for i in top_indices]
            best_labels.append(top_labels[0])

            # Log the results
            # print(f"Image {image_file} in folder '{label}' is best described by labels '{top_labels}'.")

            if label not in top_labels:
                nb_wrong_label += 1
            #     # print(f"Image {image_file} in folder '{label}' is best described by label '{best_label}'.")
            # else: #copy the image to the curated folder /dataset/train/curated/
            #     # Create the curated folder if it doesn't exist
            #     curated_folder = os.path.join(cfg.data_dir, 'train', 'curated', label)
            #     os.makedirs(curated_folder, exist_ok=True)
            #     #move the image to the curated folder
            #     os.rename(image_path, os.path.join(curated_folder, image_file))
        best_label = max(set(best_labels), key=best_labels.count)
        print(f"Number of wrong predictions for {label}: {nb_wrong_label}, better represented by {best_label}")

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
                






@hydra.main(config_path='configs/dataset_curator', config_name='config')
def export_text_from_val(cfg):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    with open(cfg.cheese_description, 'r') as file:
        cheese_description = yaml.safe_load(file)

    # Derive labels from the names of subdirectories in the specified images directory
    labels = [name for name in os.listdir(cfg.images_dir) if os.path.isdir(os.path.join(cfg.images_dir, name))]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {cheese_description[label]} cheese") for label in labels]).to(device)
    
    # Use the validation set to curate the training set
    # Process the images in the validation set
    val_input = os.path.join(cfg.data_dir, 'val')
    val_labels = [name for name in os.listdir(val_input) if os.path.isdir(os.path.join(val_input, name))]

    # Encode the validation images once, outside of the image loop
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
            # Average the image features for the label
            label_features = torch.stack(label_features).mean(dim=0)
            val_image_features.append(label_features)
        val_image_features = torch.cat(val_image_features)
    # create phrases from val_image_features
    
@hydra.main(config_path='configs/dataset_curator', config_name='config')
def curate2(cfg):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    with open(cfg.cheese_description, 'r') as file:
        cheese_description = yaml.safe_load(file)
    
    

    # Derive labels from the names of subdirectories in the specified images directory
    labels = [name for name in os.listdir(cfg.images_dir) if os.path.isdir(os.path.join(cfg.images_dir, name))]
 
    text_inputs = torch.cat([clip.tokenize(f"{label} cheese") for label in labels]).to(device)
    
    # Use the validation set to curate the training set
    # Process the images in the validation set
    val_input = os.path.join(cfg.data_dir, 'val')
    val_labels = [name for name in os.listdir(val_input) if os.path.isdir(os.path.join(val_input, name))]

    # Encode the validation images once, outside of the image loop
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
            # Average the image features for the label
            label_features = torch.stack(label_features).mean(dim=0)
            val_image_features.append(label_features)
        val_image_features = torch.cat(val_image_features)
    

    
    # Encode the text inputs once, outside of the image loop
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    # Choose text or val_image_features
    if cfg.use_val:
        features = val_image_features
    else:
        features = text_features

    # Process images within each subfolder, corresponding to their label
    nb_wrong = 0
    # print(text_inputs[0]==text_inputs[1])
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

                # Calculate cosine similarity between the image features and the val_image_features of the corresponding label
                similarity = F.cosine_similarity(image_features, features[index].unsqueeze(0)).item()
                # add to result txt
                mean_similarity += similarity

                # Optional: Determine if the similarity is below a threshold
                threshold = cfg.threshold  # Define a threshold for considering an image wrongly labeled
                if similarity < threshold:
                    if cfg.delete:
                        #delete the image
                        os.remove(image_path)
                    nb_wrong_label += 1
                    nb_wrong += 1
        print(f"Mean similarity for {label}: {mean_similarity/nb_total}")
                
        print(f"Number of wrong predictions for {label}: {nb_wrong_label} out of {nb_total}")
        results[label] = f"Number of wrong predictions for {label}: {nb_wrong_label} out of {nb_total} \n"
    #save results to a txt file
    with open("results.txt", "w") as file:
        file.write(str(results))

if __name__ == "__main__":
    curate2()