import os
import shutil
import kagglehub

def download_database(kaggle_path: str = "piyushagni5/berlin-database-of-emotional-speech-emodb", local_path: str = "~/diss/diss_autml/database"):
    try:
        # Step 1: Download the dataset
        path = kagglehub.dataset_download(kaggle_path)
        print("Downloaded dataset path:", path)

        # Step 2: Ensure the target directory exists
        local_path = os.path.expanduser(local_path)  # Expand ~ to full path
        os.makedirs(local_path, exist_ok=True)

        # Step 3: Move the dataset to the target directory
        shutil.move(path, local_path)

        print(f"Dataset successfully moved to: {local_path}")
        return local_path
    except Exception as e:
        print("Error downloading or moving dataset:", e)
        return None
