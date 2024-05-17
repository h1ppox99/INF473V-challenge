import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def fine_tune_model(cheese_type, dataset_path, output_model_path):
    """
    Fine-tune a model for a cheese type using images from dataset_path
    """
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    # Load the tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model).to(device)

    # Load the UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

    # Load the StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, unet=unet, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Define the data loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fine-tune the model
    for epoch in range(10):
        for images, labels in dataloader:
            images = images.to(device, torch.float16)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the fine-tuned model
    model.save_pretrained(output_model_path)

def dataset_path(cheese_name):
    return f"data/val/{cheese_name}"

def model_path(cheese_name):
    return f"models/{cheese_name}"

# Fine-tune the model for a specific cheese type
cheese_type = "BRIE DE MELUN"

if __name__ == "__main__":
    dataset_path_str = dataset_path(cheese_type)
    model_path_str = model_path(cheese_type)
    fine_tune_model(cheese_type, dataset_path_str, model_path_str)
