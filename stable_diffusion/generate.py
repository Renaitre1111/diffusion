import torch
import torchvision
import os
import random
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import numpy as np
from collections import Counter, defaultdict

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

config = {
    "base_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "refiner_model_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "ip_adapter_repo": "h94/IP-Adapter",
    "ip_adapter_weights_dir": "sdxl_models",
    "ip_adapter_weights_file": "ip-adapter-plus_sdxl_vit-h.safetensors"
}

def create_moduler_prompt(class_name):
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
                    for y, paths in class_to_data.items()}

    return class_to_gen, class_to_data
    
def load_generation_pipeline(config, device="cuda"):
    pipe = AutoPipelineForText2Image.from_pretrained(
        config["base_model_id"],
        torch_dtype=torch.dtype,
        variant="bf16",
        use_safetensors=True
    )

    refiner = AutoPipelineForImage2Image.from_pretrained(
        config["refiner_model_id"],
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.dtype,
        variant="bf16",
        use_safetensors=True
    )
    pipe.refiner = refiner

    pipe.load_ip_adapter(
        config["ip_adapter_repo"],
        subfolder=config["ip_adapter_weights_dir"],
        weight_name=config["ip_adapter_weights_file"],
    )
    pipe.to(device)
    return pipe

