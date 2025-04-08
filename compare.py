import os
import cv2
import numpy as np

# Define directories
test_images_dir = "dataset/test_subset_resized/images"
gt_labels_dir = "dataset/test_subset_resized/labels"
pred_labels_dir = "runs/detect/predict15/labels"
output_dir = "comparison_results"
def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Each box is (x1, y1, x2, y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def load_yolo_labels(label_path, img_width, img_height):
    """
    Load YOLO-format boxes from a label file and convert normalized coordinates to pixels.
    Returns a list of boxes as (x1, y1, x2, y2, class_id).
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes  # Return empty list if file is missing.
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            boxes.append((x1, y1, x2, y2, int(class_id)))
    return boxes

def match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth boxes.
    Each ground truth box can be matched only once.
    
    Returns:
        match_ious: list of IoU scores for matched boxes.
        match_count: number of matched boxes.
    """
    match_ious = []
    remaining_gt = gt_boxes.copy()  # To avoid matching a GT box more than once.
    for pred in pred_boxes:
        best_iou = 0.0
        best_match = None
        for gt in remaining_gt:
            iou = compute_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_match = gt
        if best_iou >= iou_threshold and best_match is not None:
            match_ious.append(best_iou)
            remaining_gt.remove(best_match)
    return match_ious, len(match_ious)

def annotate_image(image, avg_iou, match_count, gt_count, pred_count):
    """
    Annotate the image with detection statistics:
    - Good: matched / total ground truth.
    - Extra: predicted boxes not matched.
    - Missed: ground truth boxes not matched.
    - Average IoU for matched boxes.
    """
    extra = max(pred_count - match_count, 0)
    missed = max(gt_count - match_count, 0)
    
    text_good = f"Good: {match_count}/{gt_count}"
    text_wrong = f"Extra: {extra}, Missed: {missed}"
    text_iou = f"Avg IoU: {avg_iou*100:.1f}%"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color_good = (0, 255, 0)  # Green for good
    color_wrong = (0, 0, 255) # Red for wrong
    color_iou = (255, 0, 0)   # Blue for IoU
    
    cv2.putText(image, text_good, (10, 30), font, font_scale, color_good, thickness)
    cv2.putText(image, text_wrong, (10, 60), font, font_scale, color_wrong, thickness)
    cv2.putText(image, text_iou, (10, 90), font, font_scale, color_iou, thickness)
    return image

os.makedirs(output_dir, exist_ok=True)

# Process each test image
image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)
    gt_label_path = os.path.join(gt_labels_dir, os.path.splitext(image_file)[0] + ".txt")
    pred_label_path = os.path.join(pred_labels_dir, os.path.splitext(image_file)[0] + ".txt")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}. Skipping.")
        continue
    
    height, width = img.shape[:2]
    gt_boxes = load_yolo_labels(gt_label_path, width, height)
    pred_boxes = load_yolo_labels(pred_label_path, width, height)
    
    # Count objects
    gt_count = len(gt_boxes)
    pred_count = len(pred_boxes)
    
    # Match predictions (only for detected objects)
    match_ious, match_count = match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5)
    avg_iou = sum(match_ious) / len(match_ious) if match_ious else 0.0
    
    # Annotate image with detection statistics.
    annotated_img = annotate_image(img.copy(), avg_iou, match_count, gt_count, pred_count)
    
    # Optionally, draw the boxes: ground truth in green and predictions in red.
    for (x1, y1, x2, y2, _) in gt_boxes:
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x1, y1, x2, y2, _) in pred_boxes:
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, annotated_img)
    print(f"Processed {image_file}: Good {match_count}/{gt_count}, Extra {pred_count - match_count}, Missed {gt_count - match_count}, Avg IoU: {avg_iou*100:.1f}% - Saved to {output_path}")
# Initialize counters
total_gt_objects = 0
total_pred_objects = 0
total_correct_matches = 0
total_iou_sum = 0.0

# Replace the current loop with this version that updates the counters:
for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)
    gt_label_path = os.path.join(gt_labels_dir, os.path.splitext(image_file)[0] + ".txt")
    pred_label_path = os.path.join(pred_labels_dir, os.path.splitext(image_file)[0] + ".txt")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}. Skipping.")
        continue
    
    height, width = img.shape[:2]
    gt_boxes = load_yolo_labels(gt_label_path, width, height)
    pred_boxes = load_yolo_labels(pred_label_path, width, height)
    
    gt_count = len(gt_boxes)
    pred_count = len(pred_boxes)
    
    match_ious, match_count = match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5)
    avg_iou = sum(match_ious) / len(match_ious) if match_ious else 0.0
    
    # Update stats
    total_gt_objects += gt_count
    total_pred_objects += pred_count
    total_correct_matches += match_count
    total_iou_sum += sum(match_ious)
    
    # Annotate and save image
    annotated_img = annotate_image(img.copy(), avg_iou, match_count, gt_count, pred_count)
    for (x1, y1, x2, y2, _) in gt_boxes:
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x1, y1, x2, y2, _) in pred_boxes:
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, annotated_img)

# ---- Create results.txt ----
wrong_detections = total_pred_objects - total_correct_matches
missed_detections = total_gt_objects - total_correct_matches
detection_accuracy_pct = (total_correct_matches / total_gt_objects * 100) if total_gt_objects > 0 else 0
average_iou_overall = (total_iou_sum / total_correct_matches * 100) if total_correct_matches > 0 else 0

with open("results.txt", "w") as f:
    f.write("📊 Model Evaluation Summary\n")
    f.write("===========================\n")
    f.write(f"Total Ground Truth Objects : {total_gt_objects}\n")
    f.write(f"Total Predicted Objects    : {total_pred_objects}\n")
    f.write(f"Correct Detections         : {total_correct_matches}/{total_gt_objects}\n")
    f.write(f"Wrong Detections           : {wrong_detections}\n")
    f.write(f"Missed Detections          : {missed_detections}\n")
    f.write(f"Detection Accuracy         : {detection_accuracy_pct:.2f}%\n")
    f.write(f"Average IoU of Matches     : {average_iou_overall:.2f}%\n")

print("✅ Evaluation complete. Results saved to 'results.txt'")

print("Processing complete. Check the 'accuracy_comparison_stats' folder for results.")
