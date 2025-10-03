import json
import os
from openai import OpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from skimage import io


client = OpenAI()
prompt_client = OpenAI()

# Load your dataset
with open("data/data.json", "r") as f:
    data = json.load(f)

output_dir = "house_images"
os.makedirs(output_dir, exist_ok=True)

for house in data["RealEstateObj"]:
    # 1. Create a good image generation prompt with a chat model
    # Build prompt from data
    prompt = (
        f"A realistic real estate photo of a {house['Bedrooms']}-bedroom, "
        f"{house['Bathrooms']}-bathroom home in {house['Neighborhood']}, "
        f"about {house['HouseSize']} sqft. "
        f"{house['Description']} Neighborhood: {house['NeighborhoodDescription']}. "
        f"Style: professional real estate catalog photography. No text."
        f"Photo should be from outside."
    )

    system_promt = """You are a generator for DALL-E prompts. Rewrite the following house description as a short, vivid prompt for DALL·E,
               suitable for a professional real estate catalog photo. 
               Include architectural style, atmosphere, lighting, and outdoor features, but do NOT include text in the image."""

    
    # Create image prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": prompt}
        ]
    )

    image_prompt = response.choices[0].message.content


    # 2. Call DallE with langchain wrapper (OpenAI API had problems with the used api key)
    image_url = DallEAPIWrapper().run(image_prompt)
    image_data = io.imread(image_url)

    # Save image
    filename = os.path.join(output_dir, f"{house['Neighborhood'].lower()}_{house['Bedrooms']}br.png")
    io.imsave(filename, image_data)

    # Add image path to JSON entry
    house["ImagePath"] = filename

# Save updated dataset
with open("houses_with_images.json", "w") as f:
    json.dump(data, f, indent=2)
