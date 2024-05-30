import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import asyncio
import os
import random
import yaml
import aiofiles
import hydra

async def load_labels(filepath):
    labels = []
    async with aiofiles.open(filepath, 'r') as f:
        async for line in f:
            labels.append(line.strip())
    return labels

async def load_yaml(filepath):
    async with aiofiles.open(filepath, 'r') as file:
        return yaml.safe_load(await file.read())

async def load_image_paths(base_path):
    image_paths = {}
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if os.path.isdir(label_path):
            image_paths[label] = [os.path.join(label_path, image) for image in os.listdir(label_path)]
    return image_paths

async def generate_images(label, images, inpaint_pipeline, cfg, i):
    prompt = "high quality, isolated on a white background"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
    for img in images:
        img = img.convert("RGB")
        mask = Image.new("L", img.size, 0)  # Creating a mask with a black background (0)
        # Assuming the cheese is centered, create a white circle mask (255) for inpainting
        mask_draw = ImageDraw.Draw(mask)
        width, height = img.size
        radius = min(width, height) // 2
        center = (width // 2, height // 2)
        mask_draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

        result = inpaint_pipeline(prompt=prompt, image=img, mask_image=mask, num_inference_steps=50, guidance_scale=7.5).images[0]
        directory = os.path.join(cfg.output_dir, label)
        os.makedirs(directory, exist_ok=True)
        result.save(f"{directory}/{label}_{i}.png")
        print(f"Saved image: {directory}/{label}_{i}.png")

async def generate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-inpainting"
    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id).to(device)
    
    labels = await load_labels(cfg.labels_path)
    image_paths = await load_image_paths(cfg.cheese_path)

    tasks = []
    for label in labels:
        selected_image_paths = random.sample(image_paths[label], cfg.num_images_per_label)
        images = [Image.open(path) for path in selected_image_paths]
        for i in range(cfg.num_images_per_label):
            tasks.append(generate_images(label, images, inpaint_pipeline, cfg, i))

    await asyncio.gather(*tasks)

@hydra.main(config_name='config')
def main(cfg):
    asyncio.run(generate(cfg))

if __name__ == "__main__":
    main()
