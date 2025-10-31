import torch
import os
import random
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image

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

def create_moduler_prompt(class_name):
    style = random.choice(MODULE_STYLE)
    context = random.choice(MODULE_CONTEXT)
    view = random.choice(MODULE_VIEW)

    class_name_formatted = class_name.replace('_', ' ')

    prompt = f"{style} {class_name_formatted}, {view}, {context}"
    return prompt

