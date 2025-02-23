import os
import shutil
import logging
from typing import Optional

import kagglehub

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
    if os.path.exists(local_path):
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
            os.makedirs(local_path, exist_ok=True)
            logger.debug("Created directory: %s", local_path)
        except Exception as e:
            logger.error("Failed to create directory: %s", e)
            raise CreateDirError(f"Failed to create dataset directory: {e}")

        # Step 3: Move the dataset to the target directory
        try:
            shutil.move(path, local_path)
            logger.info("Dataset successfully moved to: %s", local_path)
            return local_path
        except Exception as e:
            logger.error("Failed to move dataset: %s", e)
            raise MoveError(f"Failed to move dataset to {local_path}: {e}")

    except (DownloadError, MoveError) as e:
        logger.exception("An error occurred in download_database")
        return None
