import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

class SDXLGenerator:
    def __init__(self):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            revision="fp16",  # Ensure to use the correct revision if specific needs
            
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

# Example Usage
if __name__ == "__main__":
    generator = SDXLGenerator()
    prompts = ["A fantastical landscape with mountains in the background and a clear blue sky"]
    images = generator.generate(prompts)
    #save images
    for i, image in enumerate(images):
        image.save(f"image_{i}.png")