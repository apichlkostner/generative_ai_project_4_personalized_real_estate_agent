"""
user_data.py

This module provides sample user questionnaire data for a real estate or property selection application.
It contains functions that return sets of questions and corresponding example answers representing
different user profiles and preferences.
"""

def get_info():
    questions = [   
        "How big do you want your house to be?",
        "What are 3 most important things for you in choosing this property?", 
        "Which amenities would you like?", 
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
        "How should your house look like?"
        ]
    answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters.",
        "A red house with big windows."
    ]

    return questions, answers


def get_info2():
    questions = [   
        "How big do you want your house to be?",
        "What are 3 most important things for you in choosing this property?", 
        "Which amenities would you like?", 
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
        "How should your house look like?"
    ]
    answers = [
        "A compact two-bedroom home with an open floor plan and efficient use of space.",
        "Proximity to public parks, low property taxes, and minimal maintenance requirements.",
        "A home office space, high-speed internet connectivity, and smart home features.",
        "Walking distance to a subway station, access to ride-sharing services, and EV charging stations.",
        "A vibrant urban environment with cafes, cultural venues, and nightlife within walking distance.",
        "A modern minimalist design with clean lines, neutral colors, and sustainable materials."
    ]

    return questions, answers

def get_info3():
    questions = [   
        "How big do you want your house to be?",
        "What are 3 most important things for you in choosing this property?", 
        "Which amenities would you like?", 
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
        "How should your house look like?"
    ]
    answers = [
        "A spacious four-bedroom home with a dedicated guest room and home gym area.",
        "Waterfront location, privacy from neighbors, and stunning mountain views.",
        "Swimming pool, outdoor entertainment area, and a gourmet chef's kitchen.",
        "Private boat dock, helicopter landing pad, and secure underground parking.",
        "A secluded rural setting with complete privacy and natural surroundings.",
        "A luxurious Mediterranean villa style with arched windows and terracotta roof tiles."
    ]

    return questions, answers