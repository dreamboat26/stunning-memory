import torch
from torch import autocast
# from diffusers import StableDiffusionPipeline, DDIMScheduler
import sys
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
# import torch
from lora_diffusion import monkeypatch_lora, tune_lora_scale, monkeypatch_add_lora



# if len(sys.argv) > 1:
#     model_path = sys.argv[1]
# else:
#     model_path = 'fine-tuned-model-output/800'

if len(sys.argv) > 2:
    prompt = sys.argv[2]
else:
    prompt = "a photo of avtr"

model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")



g_cuda = None
#@markdown Can set random seed here for reproducibility.
g_cuda = torch.Generator(device='cuda')
seed = 52362 #@param {type:"number"}
g_cuda.manual_seed(seed)

 #@param {type:"string"}
negative_prompt = "" #@param {type:"string"}
num_samples = 4 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 60 #@param {type:"number"}
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}

from lora_diffusion import monkeypatch_lora, tune_lora_scale


monkeypatch_lora(pipe.unet, torch.load("output_example_text/lora_weight.pt"))
monkeypatch_lora(pipe.text_encoder, torch.load("output_example_text/lora_weight.text_encoder.pt"), target_replace_module=["CLIPAttention"])
# tune_lora_scale(pipe.unet, 1.00)
# tune_lora_scale(pipe.unet, 0.9)
# tune_lora_scale(pipe.text_encoder, 0.9)


monkeypatch_add_lora(pipe.unet, torch.load("output_example/lora_weight_e599_s3000.pt"), alpha=1.0, beta = 1.0)
tune_lora_scale(pipe.unet, 0.9)
tune_lora_scale(pipe.text_encoder, 0.9)


# torch.manual_seed(0)
with autocast("cuda"), torch.inference_mode():
    images = pipe(prompt,         
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda).images
# image.save("../contents/lora_with_clip.jpg")
# image

# with autocast("cuda"), torch.inference_mode():
#     images = pipe(
#         prompt,
        # height=height,
        # width=width,
        # negative_prompt=negative_prompt,
        # num_images_per_prompt=num_samples,
        # num_inference_steps=num_inference_steps,
        # guidance_scale=guidance_scale,
        # generator=g_cuda
#     ).images
i = 0
for img in images:
    # save image to disk
    img.save("output"+str(i)+".png")
    i+=1
