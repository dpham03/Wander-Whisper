""" FlightScraper.py constants """
class FlightScraper:
    URL_AIRPORTS = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    URL_ROUTES = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

    URL_AIRPORTS_FILE_NAME = 'airports.dat'
    URL_ROUTES_FILE_NAME = 'routes.dat'
    
    DATA_CSV_FLIGHT_DETAILS_FORMAT = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "timezone", "dst", "tz", "type", "source"]
    DATA_CSV_FLIGHT_DETAILS_COLUMNS = [1, 2, 3, 4]  
    DATA_CSV_ROUTES_DETAILS_FORMAT = ["airline", "airline_id", "source_airport", "source_airport_id", "destination_airport", "destination_airport_id", "codeshare", "stops", "equipment"]
    DATA_CSV_ROUTES_DETAILS_COLUMNS = [2, 4] 
    
class HTTP:
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    BAD_REQUEST = 400    
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    INTERNAL_SERVER_ERROR = 500

class KnownDirs:
    IMAGE_DIR = "Media/images/"
    TEXT_FILE_PATH = "Media/prompt.txt"
    API_DIR = "API/"
    DUMMY_DIR = "ThisDirWillNeverExist/"

class Config:
    TOP_K = 5
    ALPHA_DEFAULT = 0.5
    BETA_DEFAULT = 0.5
    IMAGE_ONLY_AB = 1,0
    PROMPT_ONLY_AB = 0,1

class Media:
    ALLOWED_FILE_TYPES = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')