import os
import shutil
from constants import KnownDirs, Media

def createMedia():
    os.makedirs(KnownDirs.IMAGE_DIR, exist_ok=True)

def addImage(image):
    if mediaSafeCheck(image) is False:
        return False
    
    image_path = os.path.join(KnownDirs.IMAGE_DIR, image.name)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    with open(image_path, "wb+") as dest:
        for chunk in image.chunks():
            dest.write(chunk)

    return True

def addPrompt(prompt):
    try:
        os.makedirs(os.path.dirname(KnownDirs.TEXT_FILE_PATH), exist_ok=True)
        with open(KnownDirs.TEXT_FILE_PATH, "w") as f:
            f.write(prompt)
    except:
        return False
    return True

def readPrompt(base_dir, back_pedal):
    back_pedal_translated = '../' * back_pedal
    os.makedirs(os.path.dirname(KnownDirs.TEXT_FILE_PATH), exist_ok=True)
    relative_prompt_path = os.path.abspath(
        os.path.join(
                base_dir, 
                back_pedal_translated, 
                KnownDirs.API_DIR + KnownDirs.TEXT_FILE_PATH
            )
        )
    p = open(relative_prompt_path, "r")
    try:
        return p.read()
    except:
        return None

def mediaSafeCheck(image):
    if not image.name.lower().endswith(Media.ALLOWED_FILE_TYPES):
        return False
    return True

def mediaHasDataCheck():
    prompt_exists = os.path.exists(KnownDirs.TEXT_FILE_PATH)
    images_exist = os.path.exists(KnownDirs.IMAGE_DIR) and any(
        filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) for filename in os.listdir(KnownDirs.IMAGE_DIR)
    )
    return (prompt_exists, images_exist)

def cleanupImages():
    if os.path.exists(KnownDirs.IMAGE_DIR):
        shutil.rmtree(KnownDirs.IMAGE_DIR)  # Deletes all images
        os.makedirs(KnownDirs.IMAGE_DIR)  # Recreate empty folder

def cleanupPrmopt():
# Delete the prompt file if it exists
    if os.path.exists(KnownDirs.TEXT_FILE_PATH):
        os.remove(KnownDirs.TEXT_FILE_PATH)

def cleanupMedia():
    cleanupImages()
    cleanupPrmopt()
    