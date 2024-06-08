import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"



class SDXLGenerator:
    def __init__(
        self,
    ):
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        # Take 2-steps lora as an example
        ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
        # Load model.
        self.pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipe.fuse_lora()
        # Ensure ddim scheduler timestep spacing set as trailing !!!
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        # lower eta results in more detail
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