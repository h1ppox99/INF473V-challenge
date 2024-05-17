#!/usr/bin/env python
# coding=utf-8
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import AutoTokenizer, AutoModel
import logging
import copy
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from pathlib import Path

check_min_version("0.28.0.dev0")

logger = logging.getLogger(__name__)

class DreamBoothDataset(torch.utils.data.Dataset):
    def __init__(self, instance_data_root, instance_prompt, tokenizer, size=512, center_crop=False):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_data_root = instance_data_root
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        text_inputs = self.tokenizer(self.instance_prompt, padding="max_length", truncation=True, return_tensors="pt")
        example["instance_prompt_ids"] = text_inputs.input_ids
        return example

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)
    batch = {"input_ids": input_ids, "pixel_values": pixel_values}
    return batch

def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator()
    logging.basicConfig(level=logging.INFO)
    logger.info(accelerator.state)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    text_encoder = AutoModel.from_pretrained(args.pretrained_model_name_or_path).to(accelerator.device)

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(accelerator.device)
    ckpt_path = hf_hub_download(args.repo, args.ckpt)
    unet.load_state_dict(load_file(ckpt_path), strict=False)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        torch_dtype=torch.float16
    ).to(accelerator.device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Add LoRA layers to the UNet and text encoder
    unet_lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                  target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"])
    unet.add_adapter(unet_lora_config)
    
    text_lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                  target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
    text_encoder.add_adapter(text_lora_config)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipe.parameters()), lr=args.learning_rate)

    dataset = DreamBoothDataset(args.instance_data_dir, args.instance_prompt, tokenizer, size=args.resolution, center_crop=args.center_crop)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.max_train_steps if args.max_train_steps else args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=max_train_steps)

    accelerator.register_for_checkpointing(optimizer)
    accelerator.register_for_checkpointing(lr_scheduler)

    pipe, optimizer, dataloader, lr_scheduler = accelerator.prepare(pipe, optimizer, dataloader, lr_scheduler)

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    for epoch in range(args.num_train_epochs):
        pipe.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(pipe):
                pixel_values = batch["pixel_values"]
                input_ids = batch["input_ids"]
                outputs = pipe(pixel_values, input_ids=input_ids)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    if accelerator.is_main_process:
        unet_lora_state_dict = get_peft_model_state_dict(unet)
        text_encoder_lora_state_dict = get_peft_model_state_dict(text_encoder)

        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)

        unet_state_dict = {k: v for k, v in unet_lora_state_dict.items()}
        text_encoder_state_dict = {k: v for k, v in text_encoder_lora_state_dict.items()}

        LoraLoaderMixin.save_lora_weights(args.output_dir, unet_lora_layers=unet_state_dict,
                                          text_encoder_lora_layers=text_encoder_state_dict)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA for cheese classification.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--repo", type=str, required=True, help="Repository ID from huggingface.co.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file name.")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Directory containing training data for the instance images.")
    parser.add_argument("--instance_prompt", type=str, required=True, help="Prompt for the instance images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for input images.")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop input images.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--max_train_steps", type=int, help="Maximum number of training steps. Overrides num_train_epochs if provided.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, help="Model ID for the Hugging Face Hub.")
    parser.add_argument("--rank", type=int, default=4, help="The dimension of the LoRA update matrices.")

    args = parser.parse_args()
    main(args)
