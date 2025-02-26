import os
import shutil
import logging
from typing import Optional
from collections import Counter
import tools

import kagglehub
from sklearn.model_selection import train_test_split


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG during development, INFO or WARNING in production

# Create handler (you can configure different handlers later in your main app)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DownloadError(Exception):
    """Exception raised for errors in the downloading process."""
    pass

class MoveError(Exception):
    """Exception raised for errors in the moving process."""
    pass

class CreateDirError(Exception):
    """Exception raised for errors in the moving process."""
    pass

#TODO creat, exist dir, moving data could be thier own functions
def download_database(kaggle_path: str = "piyushagni5/berlin-database-of-emotional-speech-emodb", 
                      local_path: str = "~/diss/diss_autml/database") -> Optional[str]:
    
    local_path = os.path.expanduser(local_path)  # Expand ~ to full path
    if tools.dirctory_exists(local_path):
        logger.info("Folder already exists at: %s", local_path)
        return local_path
    try:
        # Step 1: Download the dataset
        try:
            path = kagglehub.dataset_download(kaggle_path)
            logger.info("Downloaded dataset from Kaggle: %s", path)
        except Exception as e:
            logger.error("Failed to download dataset: %s", e)
            raise DownloadError(f"Failed to download dataset from Kaggle: {e}")

        # Step 2: Ensure the target directory exists
        try:
            tools.create_directory(local_path)
            logger.debug("Created directory: %s", local_path)
        except Exception as e:
            logger.error("Failed to create directory: %s", e)
            raise CreateDirError(f"Failed to create dataset directory: {e}")

        # Step 3: Move the dataset to the target directory
        try:
            tools.move_file_to_target(path, local_path)
            logger.info("Dataset successfully moved to: %s", local_path)
            return local_path
        except Exception as e:
            logger.error("Failed to move dataset: %s", e)
            raise MoveError(f"Failed to move dataset to {local_path}: {e}")

    except (DownloadError, MoveError) as e:
        logger.exception("An error occurred in download_database")
        return None
    

def generate_train_test_sample(dataset_path: str = "database/1/wav",
                               test_path: str = "database/1/demo/test",
                               train_path: str = "database/1/demo/train",
                               train_test_split_ratio: float = 0.8):
    """
    Splits a dataset of audio files into training and testing sets using sklearn.

    Args:
        dataset_path (str): Path to the original dataset containing audio files.
        test_path (str): Path to store the testing set.
        train_path (str): Path to store the training set.
        train_test_split_ratio (float): Proportion of data to allocate for training (default: 0.8).

    Returns:
        None
    """
    # Ensure train and test directories exist
    tools.create_directory(train_path)
    tools.create_directory(test_path)
    # List all files in the dataset path
    files = tools.list_files(dataset_path)

    if not files:
        logger.info("No files found in the dataset directory.")
        return

    # Split files using sklearn
    train_files, test_files = train_test_split(files, train_size=train_test_split_ratio, random_state=42, shuffle=True)

    # Move files to their respective directories
    for file in train_files:
        tools.move_file_to_target(os.path.join(dataset_path, file), os.path.join(train_path, file))

    for file in test_files:
        tools.move_file_to_target(os.path.join(dataset_path, file), os.path.join(test_path, file))

    if tools.dirctory_empty(train_path) or tools.dirctory_empty(test_path):
        logger.error("Failed to split dataset into training and testing sets.")
        return
    if tools.dirctory_empty(dataset_path):
        tools.remove_directory(dataset_path)
        logger.info("Dataset directory is now removed and data moved to test and traib dirc.")

    logger.info(f"Training samples: {len(train_files)}, Testing samples: {len(test_files)}")



# def generate_metadata_for_emodb():
#     # database/1/wav
#     char_count = Counter()
#     directory = "database/1/wav"

#     for file_name in os.listdir(directory):
#         char_count[file_name[5]] += 1

#     return dict(char_count)


# def reassemble_by_emo():
#     m = {
#         'N': 'neutral',
#         'L': 'boredom',
#         'W': 'anger',
#         'E': 'disgust',
#         'F': 'happiness',
#         'T': 'sadness',
#         'A': 'fear'
#     }
#     directory = "database/1/wav"
#     target = 'database/1'
    
#     for file_name in os.listdir(directory):
#         if len(file_name) < 6 or file_name[5] not in m:
#             continue
        
#         emotion = m[file_name[5]]
#         target_dir = os.path.join(target, emotion)
#         os.makedirs(target_dir, exist_ok=True)
        
#         src_path = os.path.join(directory, file_name)
#         dst_path = os.path.join(target_dir, file_name)
#         shutil.copy2(src_path, dst_path)