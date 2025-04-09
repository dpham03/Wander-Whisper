# django tools
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from .models import EmbeddingConfig

# utilities
import json
from constants import KnownDirs, HTTP, Config
import os
import sys
    
# core functionality
from FlightScraper import SearchFlights
import MediaOperator

# debugging tools
import time

# database and embedding tools
sys.path.append('..')
from API.src.recommend import get_recommendations_with_time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@csrf_exempt
def upload_image(request):
    """ Post the images """
    if not request.method == "POST":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.BAD_REQUEST)

    images = request.FILES.getlist("image")
    if not images:
        print("No images in request.FILES")
        return JsonResponse({"error": "No image uploaded."}, status=HTTP.BAD_REQUEST)

    for image in images:
        if MediaOperator.addImage(image) is False:
            return JsonResponse({"error": "Invalid image format. Only .jpg, .jpeg, .png, .gif, and .bmp are allowed."}, status=HTTP.BAD_REQUEST)

    return JsonResponse({"success": f"Image '{len(images)}' uploaded."}, status=HTTP.CREATED)
    
@csrf_exempt
def upload_prompt(request):
    """ Post the user prompt """
    if not request.method == "POST":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    
    data = json.loads(request.body)
    prompt = data.get("prompt", "")
    if MediaOperator.addPrompt(prompt):
        return JsonResponse({"success": f"Prompt saved!"}, status=HTTP.CREATED)

    return JsonResponse({"error": "Invalid JSON format."}, status=HTTP.BAD_REQUEST)
    
@csrf_exempt
def set_alpha_beta(request):
    """ Post the alpha value (find and set the beta value) """
    if not request.method == "POST":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    try:
        data = json.loads(request.body)
        alpha = float(data.get("alpha", 0))
        beta = float(data.get("beta", 0))

        if not (alpha + beta == 1 and 0 <= alpha <= 1 and 0 <= beta <= 1):
            return JsonResponse({"error": "Alpha and Beta must sum to 1."}, status=HTTP.BAD_REQUEST)

        config, _ = EmbeddingConfig.objects.get_or_create(id=1)
        config.alpha, config.beta = alpha, beta
        config.save()

        return JsonResponse({"success": f"Alpha set to {alpha}, beta set to {beta}."}, status=HTTP.OK)
    except (ValueError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid alpha/beta values."}, status=HTTP.BAD_REQUEST)

@csrf_exempt
def reset_to_startup(request):
    if not request.method == "DELETE":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    MediaOperator.cleanupMedia()
    MediaOperator.createMedia()
    return JsonResponse({"success": "Media is reset"}, status=HTTP.ACCEPTED)

@csrf_exempt
def find_recommended_cities(request):
    # safety checks
    if not request.method == "GET":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    
    # cursory checks for populated paths
    prompt_exists, images_exist = MediaOperator.mediaHasDataCheck()
    
    # if nothing was populated, give up
    if not prompt_exists and not images_exist:
        return JsonResponse({"error": "No images or prompt to read"}, status=HTTP.BAD_REQUEST)

    # try to grab alpha beta
    a,b = Config.ALPHA_DEFAULT, Config.BETA_DEFAULT
    try:
        grabbedConfig = EmbeddingConfig.objects.first()
        if grabbedConfig:
            a,b = grabbedConfig.alpha, grabbedConfig.beta        
        if images_exist and not KnownDirs.TEXT_FILE_PATH:
            a,b = Config.IMAGE_ONLY_AB
        elif prompt_exists and not images_exist:
            a,b = Config.PROMPT_ONLY_AB
    except:
        return JsonResponse({"error": "Error fetching alpha/beta from database"}, status=HTTP.INTERNAL_SERVER_ERROR)
    
    # if the embedding extraction attempt fails, handle it gracefully
    recv_data = None
    try:
        # read from file
        prompt_str = MediaOperator.readPrompt(SCRIPT_DIR, 2)
        pass_image_path = KnownDirs.API_DIR + KnownDirs.IMAGE_DIR if images_exist else KnownDirs.DUMMY_DIR
        
        # reccomended_cities = json with city name, country, score, lat, long, descript
        get_recommendations_with_time(pass_image_path, prompt_str, a, b, Config.TOP_K)

        # load the json
        recv_data = None
        top_k_file = os.path.join(SCRIPT_DIR, '..', Config.TOP_K_FILE_LOCATION)
        with open(top_k_file, 'r') as f:
            recv_data = json.load(f)
        
        # make sure we got the data
        if recv_data is None:
            return JsonResponse({"error": "Error processing the embeddings"}, status=HTTP.INTERNAL_SERVER_ERROR)
        
    except Exception as e:
        print(e)
        return JsonResponse({"error": "Error processing the embeddings"}, status=HTTP.INTERNAL_SERVER_ERROR)
    
    # Now we are done
    MediaOperator.cleanupMedia()

    return JsonResponse(recv_data, safe=False)

@csrf_exempt
def find_airport_path(request):
    """
    Given a list of cities, return the optimal airport path between cities in sequence.
    """
    if not request.method == 'POST':
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    try:
        # Parse JSON body
        data = json.loads(request.body)
        cities = data.get('cities', [])

        if len(cities) < 2:
            return JsonResponse({"error": "At least two cities are required."}, status=HTTP.BAD_REQUEST)

        # Initialize FlightScraper logic
        graph = SearchFlights(fetch_from_web=True)

        # Find the flight path
        path_result = graph.find_path_between_multiple_cities(cities)

        # If the function returns an error string
        if isinstance(path_result, str):
            return JsonResponse({"error": path_result}, status=HTTP.NOT_FOUND)

        # Return a structured response
        return JsonResponse({"city_airport_paths": path_result}, status=HTTP.OK)
    
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON input."}, status=HTTP.BAD_REQUEST)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=HTTP.INTERNAL_SERVER_ERROR)

@csrf_exempt
def find_two_city_path(request):
    """
    Given a list of cities, find the fastest airport route from the first city to each other city.
    """
    if not request.method == 'POST':
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    try:
        # Parse JSON body
        data = json.loads(request.body)
        cities = data.get('cities', [])

        if len(cities) < 2:
            return JsonResponse({"error": "At least two cities are required."}, status=HTTP.BAD_REQUEST)
        
        # Initialize FlightScraper logic
        graph = SearchFlights(fetch_from_web=True)

        city1 = cities[0]
        tpath = {}
        for city2 in cities[1:]:
            # Find the flight path
            path_result = graph.find_path_between_cities(city1, city2)
            tpath.update( { city2 : path_result } )
            
        # Return a structured response
        return JsonResponse({"cities":tpath}, status=HTTP.OK)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON input."}, status=HTTP.BAD_REQUEST)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=HTTP.INTERNAL_SERVER_ERROR)