import os
import cv2
import numpy as np

# Define paths
image_folder = 'images'  # Folder with images
label_folder = 'labels'  # Folder with labels (same name as images, with .txt extension)
output_folder = 'labeled_images'  # Folder where annotated images will be saved
# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to draw bounding boxes on images
def draw_bboxes_on_image(image_path, label_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # Read the label file
    with open(label_path, 'r') as file:
        labels = file.readlines()

    # Loop through each label and draw bounding boxes
    for label in labels:
        # Parse the label
        class_id, x_center, y_center, width, height = map(float, label.strip().split())
        
        # Convert normalized values to pixel values
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        
        # Calculate the top-left and bottom-right corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Draw the rectangle on the image (using class_id as a label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put the class label (class index) on the image
        cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image with bounding boxes
    cv2.imwrite(output_path, image)

# Loop through all images and their corresponding label files
for image_name in os.listdir(image_folder):
    # Check if it's an image file (you can add more types if needed)
    if image_name.endswith(('.jpg', '.png', '.jpeg')):
        # Construct the full image path and label path
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, image_name.replace(image_name.split('.')[-1], 'txt'))
        
        # Check if label file exists
        if os.path.exists(label_path):
            # Output path where the annotated image will be saved
            output_path = os.path.join(output_folder, image_name)
            
            # Draw the bounding boxes and save the output
            draw_bboxes_on_image(image_path, label_path, output_path)
        else:
            print(f"Label file for {image_name} not found. Skipping this image.")

print("Finished processing images and saving labeled images.")
