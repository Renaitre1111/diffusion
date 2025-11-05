import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torchvision
import random
from PIL import Image, ImageFilter
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
import io
import numpy as np
import argparse
from collections import Counter, defaultdict
import math

MODULE_STYLE = [
    "a user generated photo of",
    "an amateur smartphone snapshot of",
    "a quick phone picture of",
    "a blurry photo of",
    "a photo taken by a person of",
    "a user-generated image of",
    "a snapshot from a phone of"
]

MODULE_CONTEXT = [
    "in a home kitchen",
    "on a wooden restaurant table",
    "in a white takeaway box",
    "on a messy plate",
    "at a dinner party",
    "on a crowded table",
    "with a blurry background",
    "in a bowl on a countertop",
    "on a serving tray"
]

MODULE_VIEW = [
    "top-down view",
    "a close-up shot",
    "viewed from the side",
    "a half-eaten portion of",
    "a full plate of",
    "a single piece of",
    "overhead shot"
]

GLOBAL_NEGATIVE_PROMPT = (
    "professional food photography, studio lighting, high quality, "
    "artistic, advertisement, 3d render, illustration, drawing, "
    "anime, logo, watermark, text, signature, unreal engine"
)

CONFIG = {
    "base_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "refiner_model_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "ip_adapter_repo": "h94/IP-Adapter",
    "ip_adapter_weights_dir": "sdxl_models",
    "ip_adapter_weights_file": "ip-adapter-plus_sdxl_vit-h.safetensors"
}

def create_modular_prompt(class_name):
    style = random.choice(MODULE_STYLE)
    context = random.choice(MODULE_CONTEXT)
    view = random.choice(MODULE_VIEW)

    class_name_formatted = class_name.replace('_', ' ')

    prompt = f"{style} {class_name_formatted}, {view}, {context}"
    return prompt

def get_data(data_dir, lb_idx_path):
    data_dir = os.path.join(data_dir, "food101")
    dset = torchvision.datasets.Food101(data_dir, split="train", download=True)
    img_files = dset._image_files                 # list[str]
    labels_all = np.asarray(dset._labels, np.int64)  # np.ndarray[int]
    classes = dset.classes                        # list[str]

    lb_idx = np.asarray(np.load(lb_idx_path), np.int64).ravel()

    class_to_data = defaultdict(list)
    for i in lb_idx:
        y = int(labels_all[i])
        class_to_data[y].append(img_files[i])

    if class_to_data:
        n_max = max(len(v) for v in class_to_data.values())
    else:
        n_max = 0

    class_to_gen = {classes[y]: (n_max - len(paths))
                    for y, paths in class_to_data.items() if (n_max - len(paths)) > 0}

    return class_to_gen, class_to_data, classes

def get_ip_adapter_images(image_paths, num_styles):
    num_select = min(len(image_paths), num_styles)
    selected_images = random.sample(image_paths, num_select)

    pil_images = [Image.open(path).convert('RGB') for path in selected_images]
    return pil_images
    
def load_generation_pipeline(config, device="cuda"):
    dtype = torch.float16

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        config["ip_adapter_repo"],
        subfolder="models/image_encoder",
        torch_dtype=dtype,
    )

    pipe = AutoPipelineForText2Image.from_pretrained(
        config["base_model_id"],
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        image_encoder=image_encoder,
    )

    refiner = AutoPipelineForImage2Image.from_pretrained(
        config["refiner_model_id"],
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True
    )
    pipe.refiner = refiner

    pipe.load_ip_adapter(
        config["ip_adapter_repo"],
        subfolder=config["ip_adapter_weights_dir"],
        weight_name=config["ip_adapter_weights_file"],
    )
    refiner.to(device)
    pipe.to(device)
    return pipe

def run_generation(pipe, class_to_gen, class_to_data, classes, args):
    if not class_to_gen:
        return
    
    name_to_idx = {name: i for i, name in enumerate(classes)}

    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    pipe.refiner.set_ip_adapter_scale(args.ip_adapter_scale)

    refiner_cutoff = args.refiner_cutoff
    num_inference_steps = args.steps

    total_generated = 0

    resample_filter = Image.Resampling.LANCZOS
    for class_name, num_to_gen in class_to_gen.items():
        print(f"Generating {num_to_gen} images for {class_name}")
        class_output_dir = os.path.join(args.output_dir, "generated", class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        class_idx = name_to_idx[class_name]
        image_paths = class_to_data[class_idx]

        for i in range(num_to_gen):
            ip_images = get_ip_adapter_images(image_paths, args.num_styles)
            prompt = create_modular_prompt(class_name)

            latents = pipe(
                prompt=prompt,
                negative_prompt=GLOBAL_NEGATIVE_PROMPT,
                ip_adapter_image=ip_images, 
                num_inference_steps=num_inference_steps,
                denoising_end=refiner_cutoff,
                output_type="latent",
                height=args.image_size,
                width=args.image_size
            ).images

            image = pipe.refiner(
                prompt=prompt,
                negative_prompt=GLOBAL_NEGATIVE_PROMPT,
                image=latents, 
                num_inference_steps=num_inference_steps,
                denoising_start=refiner_cutoff,
            ).images[0]

            # image = image.resize((args.image_size, args.image_size), resample=resample_filter)
            image_blurred = image.filter(ImageFilter.GaussianBlur(radius=0.3))

            save_path = os.path.join(class_output_dir, f"{class_name}_{i+1}.jpeg")
            image_blurred.save(save_path, format="JPEG", quality=100)
            total_generated += 1
    
    np.save(os.path.join(args.output_dir, "class_to_idx.npy"), name_to_idx, allow_pickle=True)
    print(f"Total generated: {total_generated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lb_idx_path", type=str, default="./stable_diffusion/food101/lb_labels_50_10_450_10_exp_random_noise_0.0_seed_1_idx.npy")
    parser.add_argument("--output_dir", type=str, default="./data/generated/lb_50_10")
    parser.add_argument("--num_styles", type=int, default=1)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.6)
    parser.add_argument("--refiner_cutoff", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--image_size", type=int, default=512)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_to_gen, class_to_data, classes = get_data(args.data_dir, args.lb_idx_path)

    pipe = load_generation_pipeline(CONFIG, device=device)
    run_generation(pipe, class_to_gen, class_to_data, classes, args)