import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import keras 
from keras.models import load_model
from dataSet import X_test, y_test

print(" Evaluating LSTM-DQN on Test Dataset...")

# 1. Load the trained network
main_network = load_model("lstm_dqn_model.keras")
assert main_network is not None, "Failed to load the model!"

# Ask the network to predict the test dataset
q_values = main_network.predict(X_test, batch_size=32) # type: ignore

# Pick the action with the highest Q-value for each step
predictions = np.argmax(q_values, axis=1)

# ==========================================
# TERMINAL OUTPUTS (TEXT)
# ==========================================

# Calculate overall accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\n OVERALL MODEL ACCURACY: {accuracy * 100:.2f}%")

# Calculate Precision, Recall, and F1
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=['Normal (0)', 'Intrusion (1)']))

# Calculate specific security metrics
cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel() 

fpr = fp / (fp + tn)
recall = tp / (tp + fn)

print(f"\nCRITICAL IDS METRICS:")
print(f"Detection Rate (Recall): {recall:.4f} (Goal: As close to 1.0 as possible)")
print(f"False Alarm Rate: {fpr:.4f} (Goal: As close to 0.0 as possible)")

# ==========================================
# PRESENTATION OUTPUTS (GRAPHS)
# ==========================================

# 1. Generate and Save the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Intrusion'], 
            yticklabels=['Normal', 'Intrusion'])
plt.ylabel('Actual Network Traffic')
plt.xlabel('AI Predicted Action')
plt.title(f'LSTM-DQN Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300) # Saves a high-res image for slides
print("\n📸 Saved 'confusion_matrix.png' to your folder.")
plt.show()

# 2. Generate and Save the ROC Curve
# We use the Q-value of the "Attack" action (index 1) as the confidence score
attack_confidences = q_values[:, 1]
fpr_roc, tpr_roc, _ = roc_curve(y_test, attack_confidences)
roc_auc = auc(fpr_roc, tpr_roc)

plt.figure(figsize=(8, 6))
plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Detection Rate)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300) # Saves a high-res image for slides
print(" Saved 'roc_curve.png' to your folder.")
plt.show()