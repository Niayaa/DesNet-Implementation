import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Confusion matrix data for three groups
confusion_matrix_group1 = np.array([[202, 26, 1, 1],
                                    [30, 69, 2, 0],
                                    [28, 7, 10, 0],
                                    [9, 3, 0, 0]])

confusion_matrix_group2 = np.array([[122,25,0,0],
                                    [12,67,0,0],
                                    [11,12,2,4],
                                    [5,7,2,0]])

#confusion_matrix_group3 = np.array([[100, 50, 25, 25],
#                                    [10, 90, 0, 0],
#                                    [5, 5, 35, 5],
#                                    [0, 0, 5, 5]])

# Create a figure with 3 subplots, one for each group of confusion matrix data
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Plot each confusion matrix
sns.heatmap(confusion_matrix_group1, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Confusion Matrix For Poly-crystallize')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

sns.heatmap(confusion_matrix_group2, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('Confusion Matrix For Mono-crystallize')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

#sns.heatmap(confusion_matrix_group3, annot=True, fmt="d", cmap="Blues", ax=axes[2])
#axes[2].set_title('Confusion Matrix For Both')
#axes[2].set_ylabel('True Label')
#axes[2].set_xlabel('Predicted Label')

# Adjust layout
plt.tight_layout()
plt.show()
