import json
import random

# Define city names
city_names = [
    "Paris", "New York", "Tokyo", "London", "Rome", "Barcelona", "Dubai", 
    "Sydney", "Los Angeles", "Bangkok", "Berlin", "Amsterdam", "Toronto", 
    "Singapore", "Hong Kong", "San Francisco", "Seoul", "Istanbul", "Moscow", "Cape Town"
]

# Define categories with variation
city_descriptions = [
    "A bustling metropolis with world-famous landmarks and a rich cultural scene.",
    "A coastal city known for its stunning beaches and vibrant nightlife.",
    "A historic city with ancient ruins and an old-town charm.",
    "A modern technology hub with futuristic architecture and high-speed trains.",
    "A city known for its luxury shopping, fine dining, and iconic skyline.",
    "A scenic destination with mountains, lakes, and outdoor adventures.",
    "A cultural melting pot offering diverse cuisine, art, and history.",
    "A city famous for its museums, theaters, and artistic heritage.",
]

weather_types = ["tropical", "temperate", "desert", "cold", "humid", "dry"]
landscapes = ["urban", "coastal", "mountainous", "rural", "island"]
transportation = ["public transport", "bicycle-friendly", "high-speed trains", "luxury cars"]
activities = ["sightseeing", "hiking", "scuba diving", "skiing", "shopping", "food tours", "cultural festivals"]
local_cuisines = ["French", "Japanese", "Italian", "Mediterranean", "Asian fusion", "Street food culture"]
image_folders = ["./images/paris", "./images/nyc", "./images/tokyo", "./images/london"]  # Example image paths

# Generate synthetic city data
cities_data = []

for city in city_names:
    city_data = {
        "name": city,
        "metadata": {
            "description": random.choice(city_descriptions),
            "weather": random.choice(weather_types),
            "landscape": random.choice(landscapes),
            "transportation": random.choice(transportation),
            "activities": random.sample(activities, 2),  # Select 2 random activities
            "cuisine": random.choice(local_cuisines),
        },
        "image_folder": random.choice(image_folders)  # Assign random image folder
    }
    cities_data.append(city_data)

# Save to JSON
with open("synthetic_city_database.json", "w") as f:
    json.dump(cities_data, f, indent=4)

print("✅ Synthetic city database generated and saved as 'synthetic_city_database.json'")
