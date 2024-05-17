## ne fonctionne pas encore

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

class SDXLLightiningGenerator:
    def __init__(self, fine_tuned_model_path=None):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        if fine_tuned_model_path:
            # Load the fine-tuned model
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                fine_tuned_model_path, torch_dtype=torch.float16
            ).to(device)
        else:
            # Load the base model and UNet
            unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
                device, torch.float16
            )
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                base, unet=unet, torch_dtype=torch.float16, variant="fp16"
            ).to(device)

        # Set the scheduler and progress bar configuration
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.num_inference_steps = 4
        self.guidance_scale = 2

    def generate(self, prompts):
        images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
        return images

# Example usage:
# Create a generator instance with the fine-tuned model path
fine_tuned_model_path = "path/to/your/fine-tuned-model"
generator = SDXLLightiningGenerator(fine_tuned_model_path=fine_tuned_model_path)
images = generator.generate(["A picture of a specific type of cheese"])
