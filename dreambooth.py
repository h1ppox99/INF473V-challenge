import os
import subprocess
import shutil
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig
from PIL import Image

# Step 1: Clone the PEFT repository and install dependencies
def setup_environment():
    subprocess.run(["git", "clone", "https://github.com/huggingface/peft"], check=True)
    os.chdir("peft/examples/lora_dreambooth")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    subprocess.run(["pip", "install", "git+https://github.com/huggingface/peft"], check=True)
    os.chdir("../../../")  # Return to the original directory

# Step 2: Organize images and prepare directories
def organize_images(src_dir, dest_dir, cheese_types):
    for cheese_type in cheese_types:
        instance_dir = os.path.join(dest_dir, f"{cheese_type}_instance")
        class_dir = os.path.join(dest_dir, f"{cheese_type}_class")
        os.makedirs(instance_dir, exist_ok=True)
        os.makedirs(class_dir, exist_ok=True)

        cheese_src_dir = os.path.join(src_dir, cheese_type)
        if not os.path.exists(cheese_src_dir):
            raise FileNotFoundError(f"Source directory for {cheese_type} does not exist: {cheese_src_dir}")

        # Move instance images
        for i, file_name in enumerate(os.listdir(cheese_src_dir)):
            src_file_path = os.path.join(cheese_src_dir, file_name)
            dest_file_path = os.path.join(instance_dir, f"{cheese_type}_{i}.jpg")
            shutil.copy(src_file_path, dest_file_path)

        # Move class images (if available)
        class_src_dir = os.path.join(src_dir, "class_images")
        if os.path.exists(class_src_dir):
            for i, file_name in enumerate(os.listdir(class_src_dir)):
                src_file_path = os.path.join(class_src_dir, file_name)
                dest_file_path = os.path.join(class_dir, f"class_{i}.jpg")
                shutil.copy(src_file_path, dest_file_path)

# Step 3: Train the model for each cheese type
def train_model(cheese_type, instance_dir, class_dir, output_dir, hf_token):
    model_name = "CompVis/stable-diffusion-v1-4"
    project_name = f"Dreambooth_{cheese_type}_SDXL"
    instance_prompt = f"a photo of {cheese_type} cheese"
    class_prompt = "a photo of cheese"
    
    command = [
        "accelerate", "launch", "peft/examples/lora_dreambooth/train_dreambooth.py",
        "--pretrained_model_name_or_path", model_name,
        "--instance_data_dir", instance_dir,
        "--class_data_dir", class_dir,
        "--output_dir", output_dir,
        "--train_text_encoder",
        "--with_prior_preservation", "--prior_loss_weight", "1.0",
        "--num_dataloader_workers", "1",
        "--instance_prompt", instance_prompt,
        "--class_prompt", class_prompt,
        "--resolution", "512",
        "--train_batch_size", "1",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--num_class_images", "200",
        "--use_lora",
        "--lora_r", "16",
        "--lora_alpha", "27",
        "--lora_text_encoder_r", "16",
        "--lora_text_encoder_alpha", "17",
        "--learning_rate", "1e-4",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing",
        "--max_train_steps", "800",
        "--push_to_hub",
        "--token", hf_token,
        "--repo_id", project_name
    ]

    # Run the training command
    subprocess.run(command, check=True)

# Step 4: Generate images using the fine-tuned model
def generate_images(cheese_type, output_dir, prompt, negative_prompt):
    model_name = "CompVis/stable-diffusion-v1-4"
    dtype = torch.float16
    device = "cuda"
    adapter_name = cheese_type

    # Load the model with LoRA weights
    def get_lora_sd_pipeline(ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"):
        unet_sub_dir = os.path.join(ckpt_dir, "unet")
        text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
        if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
            config = LoraConfig.from_pretrained(text_encoder_sub_dir)
            base_model_name_or_path = config.base_model_name_or_path

        if base_model_name_or_path is None:
            raise ValueError("Please specify the base model name or path")

        pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

        if os.path.exists(text_encoder_sub_dir):
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
            )

        if dtype in (torch.float16, torch.bfloat16):
            pipe.unet.half()
            pipe.text_encoder.half()

        pipe.to(device)
        return pipe

    pipe = get_lora_sd_pipeline(output_dir, base_model_name_or_path=model_name, dtype=dtype, device=device, adapter_name=adapter_name)
    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
    image.save(os.path.join(output_dir, f"{cheese_type}_generated.png"))

# Main script execution
if __name__ == "__main__":
    # Define the necessary paths and variables
    src_dir = "/path/to/cheese/images"
    dest_dir = "/path/to/organized/images"
    hf_token = "your-huggingface-token"
    cheese_types = ["Brie", "Cheddar", "Gouda"]  # Add your cheese types here

    setup_environment()
    organize_images(src_dir, dest_dir, cheese_types)

    for cheese_type in cheese_types:
        instance_dir = os.path.join(dest_dir, f"{cheese_type}_instance")
        class_dir = os.path.join(dest_dir, f"{cheese_type}_class")
        output_dir = os.path.join("/path/to/save/models", cheese_type)
        
        train_model(cheese_type, instance_dir, class_dir, output_dir, hf_token)
        
        # Generate images with the fine-tuned model
        prompt = f"a photo of {cheese_type} cheese in a rustic setting"
        negative_prompt = "low quality, blurry, unfinished"
        generate_images(cheese_type, output_dir, prompt, negative_prompt)
