import torch
from diffusers import (
    StableDiffusionTurboPipeline,
    UNet2DTurboConditionModel,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"


class SDTurboGenerator:
    def __init__(
        self,
    ):
        base = "stabilityai/stable-diffusion-turbo-base-1.0"
        repo = "ByteDance/SDTurbo-Lightning"
        ckpt = "sdturbo_lightning_4step_unet.safetensors"

        unet = UNet2DTurboConditionModel.from_config(base, subfolder="unet").to(
            device, torch.float16
        )
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        self.pipe = StableDiffusionTurboPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
            )
        self.pipe.set_progress_bar_config(disable=True)
        self.num_inference_steps = 4
        self.guidance_scale = 0

    def generate(self, prompts):
        images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
        return images