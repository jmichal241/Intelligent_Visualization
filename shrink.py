import os
import cv2

# Directories
image_dir = "dataset/train_subset/images"  # Change to the directory where your images are
label_dir = "dataset/train_subset/labels"  # Change to the directory where your labels are
output_image_dir = "dataset/train_subset_resized/images"  # Directory to save resized images
output_label_dir = "dataset/train_subset_resized/labels"  # Directory to save resized labels

# Create output folders if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Function to resize image without altering the labels
def resize_image(image_path, new_image_path, scale_factor=0.5):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Resize the image
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_image = cv2.resize(image, (new_width, new_height))
    cv2.imwrite(new_image_path, new_image)  # Save resized image

# Function to copy labels without any changes
def copy_labels(label_path, new_label_path):
    # Simply copy the label file to the new location
    with open(label_path, 'r') as file:
        labels = file.readlines()

    # Write the labels to the new label file without any changes
    with open(new_label_path, 'w') as file:
        for label in labels:
            file.write(label)  # No modifications to the labels

# Function to process all images and labels in the directories
def process_all_images_and_labels(image_folder, label_folder, output_image_folder, output_label_folder, scale_factor=0.5):
    # Loop through all images in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct full paths
            image_path = os.path.join(image_folder, filename)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_folder, label_filename)
            
            if os.path.exists(label_path):  # Ensure label file exists
                # Define output paths for resized image and labels
                new_image_path = os.path.join(output_image_folder, filename)
                new_label_path = os.path.join(output_label_folder, label_filename)
                
                # Resize image and copy labels
                resize_image(image_path, new_image_path, scale_factor)
                copy_labels(label_path, new_label_path)

# Run the processing on all images
process_all_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir)

