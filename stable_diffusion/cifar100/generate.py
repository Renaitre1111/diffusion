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
    "a blurry photo of",
    "a low-resolution photo of",
    "a poor quality image of",
    "a grainy photo of",
    "an out-of-focus snapshot of",
    "a cropped photo of",
    "a zoomed-in photo of"
]

MODULE_CONTEXT = [
    "in a natural setting",
    "with a cluttered background",
    "outdoors",
    "in a complex scene",
    "with a non-uniform background",
    "in the wild"
]

MODULE_VIEW = [
    "a side profile of",
    "a three-quarter view of",
    "a front view of",
    "a close-up of",
    "a wide shot of",
    "centered composition of",
    "an action shot of",
    "a (poorly framed photo) of",
    "a slightly blurred action shot of"
]

GLOBAL_NEGATIVE_PROMPT = (
    "oil painting, painting, drawing, illustration, cartoon, anime, 3d render, cgi, " 
    "people, person, human, portrait, face, skin, nsfw, nude, naked, "
    "blood, gore, violence, injury, "
    "text, caption, watermark, logo, signature, letters, words, "
    "deformed, mutated, extra limbs, out of frame, duplicate, "
    "**high resolution, 4k, 8k, sharp, clear, professional photography, studio lighting, product shot, "
    "clean background, simple background, plain background, white background, solid color background, isolated**"
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
    realism_prefix = "a real photo of, realistic photograph of, "
    
    parts = [f"{realism_prefix}{style} {name}", f"{view}", f"{context}"]

    return ", ".join(parts)

def get_data(data_dir, lb_idx_path):
    cifar100_images_dir = os.path.join(data_dir, "cifar100_images")
    data_dir = os.path.join(data_dir, "cifar100")
    dset = torchvision.datasets.CIFAR100(data_dir, train=True, download=True)
    labels_all = np.asarray(dset.targets, np.int64)  # np.ndarray[int]
    classes = dset.classes                        # list[str]

    img_files = []
    for i in range(len(dset.data)):
        img_array = dset.data[i] 
        label_idx = labels_all[i]
        class_name = classes[label_idx]

        img = Image.fromarray(img_array, 'RGB')
        class_dir = os.path.join(cifar100_images_dir, class_name)
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

    pil_images = []
    for path in selected_images:
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224), resample=Image.Resampling.LANCZOS)
        pil_images.append(img)
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

    pipe.safety_checker = None
    pipe.refiner = None

    pipe.load_ip_adapter(
        config["ip_adapter_repo"],
        subfolder=config["ip_adapter_weights_dir"],
        weight_name=config["ip_adapter_weights_file"],
    )
    pipe.to(device)
    return pipe

def run_generation(pipe, class_to_gen, class_to_data, classes, args):
    if not class_to_gen:
        return
    
    name_to_idx = {name: i for i, name in enumerate(classes)}

    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
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

            generator_seed = args.seed + i + (class_idx * num_to_gen)
            generator = torch.Generator(device=pipe.device).manual_seed(generator_seed)

            images = pipe(
                prompt=prompt,
                negative_prompt=GLOBAL_NEGATIVE_PROMPT,
                ip_adapter_image=ip_images, 
                num_inference_steps=num_inference_steps,
                height=args.gen_size,
                width=args.gen_size,
                generator=generator
            ).images
            
            image = images[0]
            
            image_resized = image.resize((args.image_size, args.image_size), resample=Image.Resampling.BILINEAR)

            current_blur_radius = random.uniform(args.min_blur_radius, args.max_blur_radius)
            image_blurred = image_resized.filter(ImageFilter.GaussianBlur(radius=current_blur_radius))

            buffer = io.BytesIO()
            current_jpeg_quality = random.randint(args.min_jpeg_quality, args.max_jpeg_quality)
            image_blurred.save(buffer, format="JPEG", quality=current_jpeg_quality) 
            image_with_artifacts = Image.open(buffer)

            current_noise_std = random.uniform(args.min_noise_std, args.max_noise_std)
            noise = np.random.normal(0, current_noise_std, (args.image_size, args.image_size, 3)).astype(np.int16)

            noisy = np.clip(np.array(image_with_artifacts, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            image_noisy = Image.fromarray(noisy, 'RGB')

            save_path = os.path.join(class_output_dir, f"{class_name}_{i+1}.png")
            image_noisy.save(save_path, format="PNG")
            total_generated += 1
    
    np.save(os.path.join("./data/generated/cifar100/lb_50_10", "class_to_idx.npy"), name_to_idx, allow_pickle=True)
    print(f"Total generated: {total_generated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    
    parser.add_argument("--lb_idx_path", type=str, default="./stable_diffusion/cifar100/lb_labels_50_10_450_10_pxe_noise_0.0_seed_1_idx.npy") 
    parser.add_argument("--output_dir", type=str, default="./data/generated/cifar100/lb_50_10/label") 
    
    parser.add_argument("--num_styles", type=int, default=1)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--gen_size", type=int, default=512)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=32)

    parser.add_argument("--min_blur_radius", type=float, default=0.1) 
    parser.add_argument("--max_blur_radius", type=float, default=1.2) 
    
    parser.add_argument("--min_noise_std", type=float, default=2.0)  
    parser.add_argument("--max_noise_std", type=float, default=15.0)
    
    parser.add_argument("--min_jpeg_quality", type=int, default=75)  
    parser.add_argument("--max_jpeg_quality", type=int, default=95)
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_to_gen, class_to_data, classes = get_data(args.data_dir, args.lb_idx_path)

    pipe = load_generation_pipeline(CONFIG, device=device)
    run_generation(pipe, class_to_gen, class_to_data, classes, args)