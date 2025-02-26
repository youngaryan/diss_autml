import os
import shutil
import logging

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def move_file_to_target(file_name, target_dir):
    shutil.move(file_name, target_dir)
    # logger.info("Moved file: %s to %s", file_name, target_dir)

def dirctory_exists(directory):
    return os.path.exists(directory)

def create_directory(directory, exist_ok=True):
    os.makedirs(directory, exist_ok=exist_ok)
    logger.info("Created directory: %s", directory)

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def remove_directory(directory):
    if dirctory_exists(directory):
        shutil.rmtree(directory)
        logger.info("Removed directory: %s", directory)
    logger.info("Directory does not exist: %s", directory)
    

def remove_file(file_path):
    os.remove(file_path)
    logger.info("Removed file: %s", file_path)

def dirctory_empty(directory):
    return os.listdir(directory) == []