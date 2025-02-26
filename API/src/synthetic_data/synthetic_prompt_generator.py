import random
import json
from sklearn.model_selection import train_test_split

# Define categories with great variability
budgets = [
    "under $500 total", "$500 - $1,000 total", "$1,000 - $2,000 total", "$2,000 - $5,000 total", "over $10,000 total",
    "budget travel ($20 per day)", "affordable travel ($50 per day)", "mid-range travel ($100 per day)", 
    "luxury travel ($300 per day)", "ultra-luxury travel ($1,000+ per day)", "all-inclusive package ($2,000 - $5,000 total)", 
    "budget backpacking (hostels, public transport, street food)", "business class budget ($5,000 - $10,000 total)", 
    "gap year budget (long-term travel, working holidays)", "digital nomad budget (long-term stay, co-living spaces)"
]

weather_prefs = ["hot", "cold", "tropical", "mild", "rainy", "snowy", "humid", "dry", "monsoon", "desert heat"]
durations = ["weekend trip", "one week", "two weeks", "one month", "long-term stay", "one night stay"]
companions = ["solo", "partner", "family of four", "friends", "business trip", "group tour", "traveling with a pet"]
dest_types = ["urban", "coastal", "mountainous", "desert", "rural", "tropical island", "historical city", "remote village"]
activities = ["hiking", "camping", "wildlife safari", "beach", "scuba diving", "snorkeling", "shopping", "nightlife"]
food_prefs = ["street food", "fine dining", "seafood", "vegetarian-friendly", "local delicacies", "Michelin-star restaurants"]
accommodations = ["hostel", "budget hotel", "luxury resort", "Airbnb", "private villa"]
transportation_modes = ["public transport", "rental car", "cycling", "walking", "scooter rental", "luxury chauffeur service"]
seasons = ["summer", "winter", "spring", "fall", "Christmas"]
events = ["Oktoberfest", "Carnival in Rio", "F1 races", "Coachella"]
languages = ["English-speaking", "French-speaking", "Spanish-speaking", "German-speaking", "Mandarin-speaking"]
visa_reqs = ["visa-free", "requires visa", "electronic travel authorization", "visa on arrival"]
safety_prefs = ["low-crime area", "female-friendly", "family-friendly", "politically stable", "tourist-friendly"]
accessibility = ["wheelchair access", "dietary restrictions", "child-friendly", "service dog friendly", "assistance for elderly"]
sustainability = ["eco-friendly", "carbon-neutral", "wildlife conservation", "sustainable tourism", "minimal impact travel"]
trip_intensity = ["relaxing", "moderate", "intense", "extreme adventure"]
cultural_prefs = ["traditional customs", "modern city life", "historical immersion", "indigenous culture", "expat-friendly"]
shopping_style = ["budget-friendly", "luxury designer", "local handicrafts", "tech gadgets", "fashion and apparel"]
internet_availability = ["high-speed WiFi", "moderate internet", "limited internet", "no internet", "satellite internet"]
luxury_rating = ["budget", "mid-range", "premium", "ultra-luxury"]
pet_friendly = ["yes", "no", "only small pets allowed", "dog-friendly beaches", "cat-friendly accommodations"]
wellness_activities = ["spa retreats", "yoga retreats", "meditation centers", "hot springs", "alternative healing therapies"]
adventure_level = ["low", "moderate", "high", "extreme"]
nightlife_preferences = ["quiet", "casual bars", "clubbing", "live music", "beach parties", "jazz clubs"]
currency_preferences = ["local currency", "USD", "EUR", "GBP", "cryptocurrency accepted"]
insurance_preferences = ["full travel insurance", "medical-only insurance", "cancellation protection", "adventure sports coverage"]
travel_addons = ["airport lounge access", "travel SIM card", "guided city tours", "car rental", "VIP airport service"]

# Departure Locations
departure_locations = ["New York, USA", "Los Angeles, USA", "Toronto, Canada", "London, UK", "Paris, France"]
departure_months = ["January", "February", "March", "April", "May", "June", "July", "August"]
return_months = departure_months  # Same as departure months

# Generate synthetic dataset
synthetic_data = {"data": []}

for _ in range(10000):  # Generate 10,000 samples
    departure_location = random.choice(departure_locations)
    departure_month = random.choice(departure_months)
    return_month = random.choice(return_months)
    
    while departure_month == return_month:
        return_month = random.choice(return_months)

    budget = random.choice(budgets)
    food_preference = random.choice(food_prefs)
    travel_companions = random.choice(companions)
    travel_duration = random.choice(durations)
    preferred_activities = random.choice(activities)
    destination_type = random.choice(dest_types)
    weather_preference = random.choice(weather_prefs)
    transportation_mode = random.choice(transportation_modes)
    season = random.choice(seasons)
    event_interest = random.choice(events)
    language_preference = random.choice(languages)
    visa_requirement = random.choice(visa_reqs)
    safety_preference = random.choice(safety_prefs)
    accessibility_needs = random.choice(accessibility)
    travel_theme = random.choice(["adventure", "luxury", "cultural", "romantic", "spiritual retreat", "wellness"])
    sustainability_focus = random.choice(sustainability)
    trip_intensity = random.choice(trip_intensity)
    cultural_preference = random.choice(cultural_prefs)
    shopping_style = random.choice(shopping_style)
    internet = random.choice(internet_availability)
    luxury = random.choice(luxury_rating)
    pet = random.choice(pet_friendly)
    wellness = random.choice(wellness_activities)
    adventure = random.choice(adventure_level)
    nightlife = random.choice(nightlife_preferences)
    currency = random.choice(currency_preferences)
    insurance = random.choice(insurance_preferences)
    travel_addon = random.choice(travel_addons)

    # Create the prompt
    prompt = f"I am departing from {departure_location} in {departure_month} and will return in {return_month}. "
    prompt += f"My budget is {budget}, and I prefer {food_preference}. "
    prompt += f"I will be traveling {travel_companions} for {travel_duration}, and I enjoy {preferred_activities}. "
    prompt += f"I prefer a {destination_type} destination with {weather_preference} weather. "
    prompt += f"My transportation preference is {transportation_mode}. "
    prompt += f"My trip is in {season}, and I am interested in {event_interest}. "
    prompt += f"The local language should be {language_preference}, and I need a {visa_requirement} destination. "
    prompt += f"My trip should be {trip_intensity}, and my adventure level is {adventure}. "
    prompt += f"For nightlife, I prefer {nightlife}. I prefer locations with {safety_preference} and {accessibility_needs} support. "
    prompt += f"I am interested in a {travel_theme} experience with {sustainability_focus} focus. "
    prompt += f"I will also be adding {travel_addon} to my trip."

    # Expected output
    expected_output = {
        "departure_location": departure_location,
        "departure_month": departure_month,
        "return_month": return_month,
        "budget": budget,
        "food_preference": food_preference,
        "travel_companions": travel_companions,
        "travel_duration": travel_duration,
        "preferred_activities": preferred_activities,
        "destination_type": destination_type,
        "weather_preference": weather_preference,
        "transportation_mode": transportation_mode,
        "season": season,
        "event_interest": event_interest,
        "language_preference": language_preference,
        "visa_requirement": visa_requirement,
        "safety_preference": safety_preference,
        "accessibility_needs": accessibility_needs,
    }

    synthetic_data["data"].append({"prompt": prompt, "output": expected_output})

# Save dataset
with open("./synthetic_prompts/expanded_synthetic_travel_data.json", "w") as f:
    json.dump(synthetic_data, f, indent=4)

print("✅ Synthetic travel data generated and saved!")
