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
    "in the wild",
    "with a blurry background",
    "indoors",
    "on a surface",
    "against a simple background"
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
    "(artistic:1.3), (digital art:1.2), (illustration:1.2), (painting:1.2), (oil painting:1.2), "
    "drawing, cartoon, anime, 3d render, cgi, "
    "concept art, artstation, deviantart, stylized, abstract, "
    "(sharp:1.2), (clear:1.2), (high resolution:1.2), (4k:1.2), (8k:1.2), "
    "professional photography, studio lighting, product shot, "
    "beautiful, perfect, aesthetic, flawless, stunning, "
    "nsfw, nude, naked, "
    "blood, gore, violence, injury, "
    "text, caption, watermark, logo, signature, letters, words, "
    "deformed, mutated, extra limbs, out of frame, duplicate"
)

CONFIG = {
    "base_model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
    "refiner_model_id": None,
    "ip_adapter_repo": "h94/IP-Adapter",
    "ip_adapter_weights_dir": "models",
    "ip_adapter_weights_file": "ip-adapter-plus_sd15.safensors"
}

def create_modular_prompt(class_name):
    style = random.choice(MODULE_STYLE)
    context = random.choice(MODULE_CONTEXT)
    view = random.choice(MODULE_VIEW)

    name = class_name.replace("_", " ")
    realism_prefix = "a real photo of, realistic photograph of, "
    
    parts = [f"{realism_prefix}{style} {name}", f"{view}", f"{context}"]

    return ", ".join(parts)

def get_data(data_dir, lb_idx_path, num_per_class):
    cifar100_images_dir = os.path.join(data_dir, "cifar100_images")
    data_dir = os.path.join(data_dir, "cifar100")
    dset = torchvision.datasets.CIFAR100(data_dir, train=True, download=True)
    labels_all = np.asarray(dset.targets, np.int64)
    classes = dset.classes

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

    class_to_gen = {classes[y]: num_per_class
                    for y in class_to_data.keys()}

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
        # variant="fp16",
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
        print("No images to generate.")
        return
    
    name_to_idx = {name: i for i, name in enumerate(classes)}

    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    num_inference_steps = args.steps
    bs = args.batch_size

    total_generated = 0

    resample_filter = Image.Resampling.BILINEAR
    
    for class_name, num_to_gen in class_to_gen.items():
        if num_to_gen <= 0:
            continue
        print(f"Generating {num_to_gen} images for {class_name}")
        class_output_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        class_idx = name_to_idx[class_name]
        image_paths = class_to_data[class_idx]

        num_batches = math.ceil(num_to_gen / bs)
        gen_count = 0

        for batch_idx in range(num_batches):
            current_bs = min(bs, num_to_gen - gen_count)
            if current_bs <= 0:
                break
            
            ip_images = get_ip_adapter_images(image_paths, args.num_styles)
            prompts = [create_modular_prompt(class_name) for _ in range(current_bs)]
            
            generator_seed = args.seed + batch_idx + (class_idx * num_batches)
            generator = torch.Generator(device=pipe.device).manual_seed(generator_seed)

            images = pipe(
                prompt=prompts,
                negative_prompt=[GLOBAL_NEGATIVE_PROMPT] * current_bs,
                ip_adapter_image=ip_images, 
                num_inference_steps=num_inference_steps,
                height=args.gen_size,
                width=args.gen_size,
                generator=generator, 
                guidance_scale=args.guidance_scale
            ).images

            for i, image in enumerate(images):
                image_resized = image.resize((args.image_size, args.image_size), resample=resample_filter)
            
                img_idx = gen_count + i + 1
                save_path = os.path.join(class_output_dir, f"{class_name}_{img_idx}.png")
                image_resized.save(save_path, format="PNG")

            gen_count += len(images)
            total_generated += len(images)

    print(f"Total generated: {total_generated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    
    parser.add_argument("--lb_idx_path", type=str, default="./stable_diffusion/cifar100/lb_labels_50_10_450_10_pxe_noise_0.0_seed_1_idx.npy") 
    parser.add_argument("--output_dir", type=str, default="./data/generated/cifar100/lb_50_10/pool") 
    
    parser.add_argument("--num_styles", type=int, default=1)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_GArgument("--gen_size", type=int, default=512)

    parser.add_argument("--image_size", type=int, default=32)

    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=30, help="Number of images to generate in a batch")
    parser.add_argument("--num_per_class", type=int, default=450, help="Fixed number of images to generate per class")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_to_gen, class_to_data, classes = get_data(args.data_dir, args.lb_idx_path, args.num_per_class)

    pipe = load_generation_pipeline(CONFIG, device=device)
    run_generation(pipe, class_to_gen, class_to_data, classes, args)