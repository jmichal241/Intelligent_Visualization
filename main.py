import os
import random
import shutil

# Set the paths to your original and new directories
dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train")
valid_dir = os.path.join(dataset_dir, "valid")
test_dir = os.path.join(dataset_dir, "test")

# New directories to store selected images
train_subset_dir = os.path.join(dataset_dir, "train_subset")
valid_subset_dir = os.path.join(dataset_dir, "valid_subset")
test_subset_dir = os.path.join(dataset_dir, "test_subset")

# Create new directories if they don't exist
os.makedirs(os.path.join(train_subset_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_subset_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(valid_subset_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(valid_subset_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(test_subset_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_subset_dir, "labels"), exist_ok=True)

# Function to copy random files
def copy_random_files(src_images, src_labels, dest_images, dest_labels, num_files):
    # Get all the image files in the source folder
    image_files = [f for f in os.listdir(src_images) if f.endswith(".jpg")]
    
    # Randomly select 'num_files' files
    selected_files = random.sample(image_files, num_files)

    # Copy selected images and labels to the destination
    for file in selected_files:
        # Copy images
        shutil.copy(os.path.join(src_images, file), os.path.join(dest_images, file))
        # Copy corresponding labels
        label_file = file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(src_labels, label_file), os.path.join(dest_labels, label_file))

# Copy random 300 images for training, 50 for validation and 50 for testing
copy_random_files(os.path.join(train_dir, "images"), os.path.join(train_dir, "labels"),
                  os.path.join(train_subset_dir, "images"), os.path.join(train_subset_dir, "labels"), 300)

copy_random_files(os.path.join(valid_dir, "images"), os.path.join(valid_dir, "labels"),
                  os.path.join(valid_subset_dir, "images"), os.path.join(valid_subset_dir, "labels"), 50)

copy_random_files(os.path.join(test_dir, "images"), os.path.join(test_dir, "labels"),
                  os.path.join(test_subset_dir, "images"), os.path.join(test_subset_dir, "labels"), 50)

print("Random subsets have been selected and copied!")


