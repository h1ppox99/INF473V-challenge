############################################
### Génération d'images avec IPAdapter   ###
###  Voir le README pour les détails     ###
############################################
import torch
from diffusers import (StableDiffusionPipeline, 
                       StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, 
                       DDIMScheduler, AutoencoderKL, StableDiffusionXLPipeline)
from PIL import Image
import yaml
import hydra
import os
from ip_adapter import IPAdapterPlusv2
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles
import random

'''
Nous avons adapté l'implémentation de IPAdapter.

'''

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

async def load_config(filepath):
    return await load_yaml(filepath)

from PIL import Image, ImageOps


async def generate_images(label, fromages, prompts, images, ip_model, cfg, negative_prompt, i):
    nb_prompts = len(prompts)
    
    for j in range(nb_prompts):
        prompt = prompts[j]
        print(f"Generating {label} with prompt: {prompt}")
        images_generes = ip_model.generate2(
            pil_image=images, num_samples=1, num_inference_steps=cfg.inference_step, seed=42,
            prompt=prompt, negative_prompt=negative_prompt, scale=cfg.scale, guidance_scale=cfg.guidance_scale,
        )
        for k, img in enumerate(images_generes):
            directory = os.path.join(cfg.output_dir, label)
            os.makedirs(directory, exist_ok=True)
            #enregistrer l'image
            img.save(f"{directory}/{label}__{i}_{j}_{k}.png")
            print("saved image number ", i, j, k)
    


async def generate(cfg):
    base_model_path = cfg.base_model_path
    vae_model_path = cfg.vae_model_path
    image_encoder_path = cfg.image_encoder_path
    ip_ckpt = cfg.ip_ckpt
    device = "cuda"
    num_images_per_label = cfg.num_images_per_label

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # charger le modèle de base
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    
    # charger le modèle IPAdapterPlusv2 (voir le readme pour la modification)
    ip_model = IPAdapterPlusv2(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
    
    labels = await load_labels(cfg.labels)
    config = await load_config(cfg.contexts_path)
    fonds = config['fonds']
    contexts = config['contexts']
    cadrages = config['cadrages']
    nb_prompts = len(contexts) * len(fonds) * len(cadrages)

    fromages = await load_yaml(cfg.cheese_description)
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    image_paths = await load_image_paths(cfg.cheese_path)

    start = time.time()

    tasks = []
    
    for label in labels:
        for i in range(num_images_per_label):
            fond = random.choice(list(fonds.values()))
            context = random.choice(list(contexts.values()))
            cadrage = random.choice(list(cadrages.values()))
            prompt = f"best quality,high quality, {context}, {fond}, {fromages[label]} "
            
            
            # Choisir 2 images aléatoires pour le label
            selected_image_paths = random.sample(image_paths[label], 2)
            images = [Image.open(path) for path in selected_image_paths]
            tasks.append(generate_images(label, fromages, [prompt], images, ip_model, cfg, negative_prompt,i))

    await asyncio.gather(*tasks)
    
    end = time.time()
    async with aiofiles.open("time.txt", "w") as f:
        await f.write(f"Time taken: {(end-start)/60} minutes")

@hydra.main(config_path="configs/IPAdapter",config_name='config')
def main(cfg):
    asyncio.run(generate(cfg))

if __name__ == "__main__":
    main()
