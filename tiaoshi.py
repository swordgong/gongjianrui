from diffusers import StableDiffusionPipeline
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

print("Pipeline loaded successfully!")