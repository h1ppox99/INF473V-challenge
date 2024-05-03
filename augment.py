import hydra
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import os
import numpy as np
from pathlib import Path

@hydra.main(config_path='configs/augment', config_name='config')
def augment(cfg):
    labels = [name for name in os.listdir(cfg.images_dir) if os.path.isdir(os.path.join(cfg.images_dir, name))]

    for label in labels:
        subfolder_path = Path(cfg.images_dir) / label
        image_files = list(subfolder_path.glob('*.jpg'))  # Assuming JPEG images
        suboutput_path = Path(cfg.save_dir) / label
        suboutput_path.mkdir(parents=True, exist_ok=True)
        for image_file in image_files:
            original_image = Image.open(image_file)
            image_np = np.array(original_image)  # Convert PIL image to numpy array

            for aug in cfg.augmentations:
                aug_obj = getattr(A, aug['type'])(**{k: v for k, v in aug.items() if k != 'type'})
                augmented_image = aug_obj(image=image_np)['image']

                # Ensure the image data type is uint8
                if augmented_image.dtype != np.uint8:
                    augmented_image = (augmented_image * 255).astype(np.uint8)

                # Save each augmented image
                output_path = suboutput_path / f"{image_file.stem}_{aug['type']}.jpg"
                Image.fromarray(augmented_image).save(output_path)

    print("Augmentation completed for all images.")

if __name__ == "__main__":
    augment()
