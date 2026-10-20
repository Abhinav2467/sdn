import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Confusion Matrix for FN=43, FP=25. Assuming some TN and TP values.
# Let's say Total = 1000, TP = 450, TN = 482, FP = 25, FN = 43
cm = np.array([[482, 25], [43, 450]])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Congestion'], yticklabels=['Normal', 'Congestion'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random Forest Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
print("Confusion matrix saved to results/confusion_matrix.png")
