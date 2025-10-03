import json
import os
from openai import OpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from skimage import io
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from diffusers.utils import load_image, make_image_grid
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

client = OpenAI()
prompt_client = OpenAI()

# Load your dataset
with open("data/data.json", "r") as f:
    data = json.load(f)

output_dir = "house_images"
os.makedirs(output_dir, exist_ok=True)

for idx, house in enumerate(data["RealEstateObj"]):
    # 1. Create a good image generation prompt with a chat model
    # Build prompt from data
    prompt = (
        f"A realistic real estate photo (exterior) of a {house['Bedrooms']}-bedroom, "
        f"{house['Bathrooms']}-bathroom home in {house['Neighborhood']}, "
        f"about {house['HouseSize']} sqft. "
        f"{house['Description']} Neighborhood: {house['NeighborhoodDescription']}. "
    )

    system_promt = """You are a generator for image generation prompts. Rewrite the following house description as a short vivid prompt,
               suitable for a professional real estate catalog photo. The houses should have different colors and styles. The prompt must be short
               so it fits the 77 token limit of CLIP."""

    
    # Create image prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": prompt}
        ]
    )

    image_prompt = response.choices[0].message.content


    # 2. Call stable diffusion model

    pipe = StableDiffusionPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        guidance_scale=12,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    image = pipe(image_prompt).images[0]

    # Save image
    filename = os.path.join(output_dir, f"{idx}.png")
    image.save(filename)

    # Add image path to JSON entry
    house["ImagePath"] = filename

# Save updated dataset
with open("houses_with_images.json", "w") as f:
    json.dump(data, f, indent=2)
