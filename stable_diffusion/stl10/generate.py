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
    "a realistic photograph of",
    "a natural photo of",
    "a high-quality photo of",
    "a detailed photo of",
    "a documentary-style photo of",
    "an outdoor photograph of"
]

MODULE_CONTEXT = [
    "in natural daylight",
    "outdoors in a realistic setting",
    "with a clean background",
    "on a simple background",
    "in its natural habitat",
    "on an open road",
    "on a runway",
    "at sea",
    "in the sky",
    "in a field"
]

MODULE_VIEW = [
    "a side profile of",
    "a three-quarter view of",
    "a front view of",
    "a close-up of",
    "a wide shot of",
    "centered composition of",
    "an action shot of"
]

GLOBAL_NEGATIVE_PROMPT = (
    "people, person, human, portrait, face, skin, nsfw, nude, naked, "
    "blood, gore, violence, injury, "
    "text, caption, watermark, logo, signature, letters, words, "
    "drawing, illustration, painting, cartoon, anime, 3d render, cgi, "
    "low quality, lowres, jpeg artifacts, blurry, deformed, mutated, extra limbs, "
    "out of frame, cropped, frame, border, duplicate, worst quality"
)

CONFIG = {
    "base_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "refiner_model_id": None,
    "ip_adapter_repo": "h94/IP-Adapter",
    "ip_adapter_weights_dir": "models",
    "ip_adapter_weights_file": "ip-adapter-plus_sd15.safetensors"
}

def create_modular_prompt(class_name):
    style = random.choice(MODULE_STYLE)
    context = random.choice(MODULE_CONTEXT)
    view = random.choice(MODULE_VIEW)

    name = class_name.replace("_", " ")
    wildlife_classes = {"bird", "cat", "deer", "dog", "horse", "monkey"}
    vehicle_classes  = {"airplane", "car", "ship", "truck"}
    extra_tag = ""
    if name in wildlife_classes:
        extra_tag = "wildlife photography"
    elif name in vehicle_classes:
        extra_tag = "transportation photography"

    parts = [f"{style} {name}", f"{view}", f"{context}"]
    if extra_tag:
        parts.append(extra_tag)

    return ", ".join(parts)

def get_data(data_dir, lb_idx_path):
    stl10_images_dir = os.path.join(data_dir, "stl10_images")
    data_dir = os.path.join(data_dir, "stl10")
    dset = torchvision.datasets.STL10(data_dir, split="train", download=True)
    labels_all = np.asarray(dset.labels, np.int64)  # np.ndarray[int]
    classes = dset.classes                        # list[str]

    img_files = []
    for i in range(len(dset.data)):
        img_array = dset.data[i] # (3, 96, 96)
        label_idx = labels_all[i]
        class_name = classes[label_idx]

        img_array = np.transpose(img_array, (1, 2, 0))
        img = Image.fromarray(img_array, 'RGB')
        class_dir = os.path.join(stl10_images_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"{class_name}_{i}.png")
        img_files.append(save_path)
        if not os.path.exists(save_path):
            img.save(save_path, "PNG")

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
    '''
    refiner = AutoPipelineForImage2Image.from_pretrained(
        config["refiner_model_id"],
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True
    )
    '''
    pipe.safety_checker = None
    pipe.refiner = None

    pipe.load_ip_adapter(
        config["ip_adapter_repo"],
        subfolder=config["ip_adapter_weights_dir"],
        weight_name=config["ip_adapter_weights_file"],
    )
    # refiner.to(device)
    pipe.to(device)
    return pipe

def run_generation(pipe, class_to_gen, class_to_data, classes, args):
    if not class_to_gen:
        return
    
    name_to_idx = {name: i for i, name in enumerate(classes)}

    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    # pipe.refiner.set_ip_adapter_scale(args.ip_adapter_scale)

    # refiner_cutoff = args.refiner_cutoff
    num_inference_steps = args.steps

    total_generated = 0

    resample_filter = Image.Resampling.LANCZOS
    for class_name, num_to_gen in class_to_gen.items():
        print(f"Generating {num_to_gen} images for {class_name}")
        class_output_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        class_idx = name_to_idx[class_name]
        image_paths = class_to_data[class_idx]

        for i in range(num_to_gen):
            ip_images = get_ip_adapter_images(image_paths, args.num_styles)
            prompt = create_modular_prompt(class_name)

            images = pipe(
                prompt=prompt,
                negative_prompt=GLOBAL_NEGATIVE_PROMPT,
                ip_adapter_image=ip_images, 
                num_inference_steps=num_inference_steps,
                height=args.gen_size,
                width=args.gen_size
            ).images
            '''
            image = pipe.refiner(
                prompt=prompt,
                negative_prompt=GLOBAL_NEGATIVE_PROMPT,
                image=latents, 
                num_inference_steps=num_inference_steps,
                denoising_start=refiner_cutoff,
            ).images[0]
            '''
            image = images[0]
            # image = image.resize((args.image_size, args.image_size), resample=resample_filter)

            image_resized = image.resize((args.image_size, args.image_size), resample=Image.Resampling.LANCZOS)

            image_blurred = image_resized.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))

            noise = np.random.normal(0, args.noise_std, (args.image_size, args.image_size, 3)).astype(np.int16)
            noisy = np.clip(np.array(image_blurred, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            image_noisy = Image.fromarray(noisy, 'RGB')

            save_path = os.path.join(class_output_dir, f"{class_name}_{i+1}.png")
            image_noisy.save(save_path, format="PNG")
            total_generated += 1
    
    np.save(os.path.join(args.output_dir, "class_to_idx.npy"), name_to_idx, allow_pickle=True)
    print(f"Total generated: {total_generated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lb_idx_path", type=str, default="./stable_diffusion/stl10/lb_labels_500_100_None_None_None_noise_0.0_seed_1_idx.npy")
    parser.add_argument("--output_dir", type=str, default="./data/generated/stl10/lb_500_100/label")
    parser.add_argument("--num_styles", type=int, default=1)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.6)
    # parser.add_argument("--refiner_cutoff", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--gen_size", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--blur_radius", type=float, default=0.4)
    parser.add_argument("--noise_std", type=int, default=2)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_to_gen, class_to_data, classes = get_data(args.data_dir, args.lb_idx_path)

    pipe = load_generation_pipeline(CONFIG, device=device)
    run_generation(pipe, class_to_gen, class_to_data, classes, args)