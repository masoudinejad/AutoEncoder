import os
import random
import shutil

import pandas as pd


def reorganize_images(
    original_dir,
):
    """
    Reorganize images into a format compatible with torchvision.datasets.ImageFolder.

    Args:
    - original_dir (str): The directory where the images and the csv file are located.
    - csv_filename (str): The name of the CSV file. Defaults to 'labels.csv'.
    """
    # Path to the CSV file
    csv_filename = "grades_mapping.csv"
    csv_path = os.path.join(original_dir, csv_filename)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # New directory for the reorganized dataset
    new_dir = os.path.join(original_dir, "reorganized")
    os.makedirs(new_dir, exist_ok=True)

    # Loop over the DataFrame and move each file into the corresponding new directory
    for index, row in df.iterrows():
        # Original file path
        original_file_path = os.path.join(original_dir, row["Patch_Filename"])

        label_dir = os.path.join(new_dir, f"grade_{row['Grade']}")
        os.makedirs(label_dir, exist_ok=True)

        # New file path in the label directory
        new_file_path = os.path.join(label_dir, row["Patch_Filename"])

        # Copy the file to the new directory
        shutil.copy2(original_file_path, new_file_path)

    print("Reorganization complete. Images are now in:", new_dir)


def create_test_dataset(input_folder, num, name):
    # Define the path for the new 'test' folder
    test_folder = os.path.join(os.path.dirname(input_folder), f"../../test_{name}")

    # Create the 'test' directory if it doesn't exist
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Iterate through each subfolder in the input folder
    for subdir in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subdir)

        # Check if it's indeed a directory
        if os.path.isdir(subfolder_path):
            # Prepare corresponding subfolder in 'test' directory
            test_subfolder_path = os.path.join(test_folder, subdir)
            os.makedirs(test_subfolder_path, exist_ok=True)

            # Get a list of images in the current subfolder
            images = [
                f
                for f in os.listdir(subfolder_path)
                if os.path.isfile(os.path.join(subfolder_path, f))
            ]

            # Randomly select 'num' images
            selected_images = random.sample(images, min(len(images), num))

            # Copy each selected image to the 'test' subfolder
            for img in selected_images:
                shutil.copy2(
                    os.path.join(subfolder_path, img),
                    os.path.join(test_subfolder_path, img),
                )

    print(f"Test dataset created at: {test_folder}")


if __name__ == "__main__":
    train_no = "./data/cell/patch_new/patch_train_no_bg/nuclei_patches"
    valid_no = "./data/cell/patch_new/patch_valid_no_bg/nuclei_patches"
    train_wt = "./data/cell/patch_new/patch_train_wt_bg/nuclei_patches_bg"
    valid_wt = "./data/cell/patch_new/patch_valid_wt_bg/nuclei_patches_bg"
    for folder_path in [train_no, valid_no, train_wt, valid_wt]:
        reorganize_images(folder_path)
    # make test subset from validation
    num_samples = 10
    validation_folder_reorganized = os.path.join(valid_no, "reorganized")
    create_test_dataset(validation_folder_reorganized, num_samples, "no_bg")
    validation_folder_reorganized = os.path.join(valid_wt, "reorganized")
    create_test_dataset(validation_folder_reorganized, num_samples, "wt_bg")
