# Personalized Real Estate Agent

For Udacity course "Generative AI"

## Functionality

- Customer answers questions about the house
- LLM creates prompts to search matching text and image descriptions
- Best matching texts and images are searched with a vector database
- The results from text and image search are combined
- The three best results are processes with a LLM again to create an individual description for the customer

## Data creation

- A LLM creates real estate datasets from an example
- A LLM creates images from the individual datasets
- A vector database is used to add the text and images

## Usage

Run server.py and open the local ip address. Fill out the questionair and press "Find Recommendations".

![alt questionair](images/questionair.png)

# House Preference Questionnaires

## Dataset 1: Family Suburban Preference

| Question | Answer |
|----------|--------|
| How big do you want your house to be? | A comfortable three-bedroom house with a spacious kitchen and a cozy living room. |
| What are 3 most important things for you in choosing this property? | A quiet neighborhood, good local schools, and convenient shopping options. |
| Which amenities would you like? | A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system. |
| Which transportation options are important to you? | Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads. |
| How urban do you want your neighborhood to be? | A balance between suburban tranquility and access to urban amenities like restaurants and theaters. |
| How should your house look like? | A red house with big windows. |

![alt result 1](images/request_1.png)

## Dataset 2: Urban Professional Preference

| Question | Answer |
|----------|--------|
| How big do you want your house to be? | A compact two-bedroom home with an open floor plan and efficient use of space. |
| What are 3 most important things for you in choosing this property? | Proximity to public parks, low property taxes, and minimal maintenance requirements. |
| Which amenities would you like? | A home office space, high-speed internet connectivity, and smart home features. |
| Which transportation options are important to you? | Walking distance to a subway station, access to ride-sharing services, and EV charging stations. |
| How urban do you want your neighborhood to be? | A vibrant urban environment with cafes, cultural venues, and nightlife within walking distance. |
| How should your house look like? | A modern minimalist design with clean lines, neutral colors, and sustainable materials. |

![alt result 2](images/request_2.png)

## Dataset 3: Luxury Estate Preference

| Question | Answer |
|----------|--------|
| How big do you want your house to be? | A spacious four-bedroom home with a dedicated guest room and home gym area. |
| What are 3 most important things for you in choosing this property? | Waterfront location, privacy from neighbors, and stunning mountain views. |
| Which amenities would you like? | Swimming pool, outdoor entertainment area, and a gourmet chef's kitchen. |
| Which transportation options are important to you? | Private boat dock, helicopter landing pad, and secure underground parking. |
| How urban do you want your neighborhood to be? | A secluded rural setting with complete privacy and natural surroundings. |
| How should your house look like? | A luxurious Mediterranean villa style with arched windows and terracotta roof tiles. |


![alt result 3](images/request_3.png)