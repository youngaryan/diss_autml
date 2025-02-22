import os
import shutil
import kagglehub

class DownloadError(Exception):
    """Exception raised for errors in the downloading process."""
    pass

class MoveError(Exception):
    """Exception raised for errors in the moving process."""
    pass

def download_database(kaggle_path: str = "piyushagni5/berlin-database-of-emotional-speech-emodb", 
                      local_path: str = "~/diss/diss_autml/database") -> str:
    try:
        # Step 1: Download the dataset
        try:
            path = kagglehub.dataset_download(kaggle_path)
            print("Downloaded dataset path:", path)
        except Exception as e:
            raise DownloadError(f"Failed to download dataset from Kaggle: {e}")

        # Step 2: Ensure the target directory exists
        local_path = os.path.expanduser(local_path)  # Expand ~ to full path
        os.makedirs(local_path, exist_ok=True)

        # Step 3: Move the dataset to the target directory
        try:
            shutil.move(path, local_path)
            print(f"Dataset successfully moved to: {local_path}")
            return local_path
        except Exception as e:
            raise MoveError(f"Failed to move dataset to {local_path}: {e}")

    except (DownloadError, MoveError) as e:
        print("Error:", e)
        return None
