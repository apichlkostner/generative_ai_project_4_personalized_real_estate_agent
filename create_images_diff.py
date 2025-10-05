"""
create_images_diff.py

This script generates images for real estate listings using Stable Diffusion.
It reads real estate data from a JSON file, generates a short vivid prompt for each listing using a language model,
and then uses Stable Diffusion to create and save an image for each listing. The image path is added to the dataset.
"""

import json
import os
from openai import OpenAI
from diffusers import StableDiffusionPipeline
import torch

from logger_config import Logger
logger = Logger(name="CreateImages").get_logger()

def main():
    logger.info("Start CreateImages")

    logger.info("Create OpenAI client")
    client = OpenAI()

    # Load your dataset
    with open("data/data.json", "r") as f:
        data = json.load(f)

    output_dir = "house_images"
    os.makedirs(output_dir, exist_ok=True)

    for idx, house in enumerate(data["RealEstateObj"]):
        logger.info(f"Create prompt for {house["Neighborhood"]}")

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

        # Generate a concise, vivid prompt for image generation using a chat model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_promt},
                {"role": "user", "content": prompt}
            ]
        )

        image_prompt = response.choices[0].message.content

        # 2. Call stable diffusion model to generate the image
        logger.info(f"Create image for {house["Neighborhood"]}")
        logger.debug(f"Prompt for {house["Neighborhood"]}: {image_prompt}")
        pipe = StableDiffusionPipeline.from_pretrained(
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        image = pipe(image_prompt).images[0]

        # Save image to disk
        filename = os.path.join(output_dir, f"{idx}.png")
        logger.info(f"Write image as {filename}")
        image.save(filename)

    logger.info("Finished CreateImages")

if __name__ == '__main__':
    main()