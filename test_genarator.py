from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch

pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-xl-v2-turbo', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "A photo of Pouligny-Saint-Pierre, a soft goat's milk cheese, featuring an elongated pyramid trunk shape with a marbled, vermiculated white-ivory crust, and a smooth, homogeneous white-ivory paste"

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=6, guidance_scale=2).images[0]  
image.save("./Pouligny.png")

