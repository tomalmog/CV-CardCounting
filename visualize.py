import pandas as pd
import matplotlib.pyplot as plt

# load and clean
df = pd.read_csv('runs/detect/train5/results.csv')
df.columns = df.columns.str.strip()  # remove leading/trailing spaces

# plot training and validation losses
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
plt.plot(df['epoch'], df['val/box_loss'], '--', label='Val Box Loss')
plt.plot(df['epoch'], df['val/cls_loss'], '--', label='Val Cls Loss')
plt.plot(df['epoch'], df['val/dfl_loss'], '--', label='Val DFL Loss')
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# plot metrics
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.show()
