import pandas as pd
import matplotlib.pyplot as plt

# Load training results
log_path = "runs/detect/train12/results.csv"  # Change this if needed
df = pd.read_csv(log_path)

# Plot loss
plt.figure(figsize=(10,5))
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='blue')
plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='red')
plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', color='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.legend()
plt.savefig("loss_plot.png")  # Save instead of showing
plt.close()  # Close the figure

# Plot precision, recall, and mAP (Mean Average Precision)
plt.figure(figsize=(10,5))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='red')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='green')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='purple')
plt.xlabel("Epoch")
plt.ylabel("Metric Score")
plt.title("Model Performance Over Time")
plt.legend()
plt.savefig("performance_plot.png")  # Save instead of showing
plt.close()  # Close the figure

print("Plots saved as loss_plot.png and performance_plot.png")

