import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

from dataset import IoTDataset
from agent import DRQN

print("🔍 Evaluating LSTM-DQN on Test Dataset...")

# 1. Load the Test Data 
# (Assuming your IoTDataset class handles a train/test split, 
# or you pass a separate test CSV here)
test_data_manager = IoTDataset('CICIOT2023_Test.csv')
X_test, y_test = test_data_manager.get_data()

# 2. Initialize and Load the Trained PyTorch Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_test.shape[2] # Number of features

model = DRQN(input_dim, 128, 2).to(device)
model.load_state_dict(torch.load("lstm_dqn_model.pth"))
model.eval() # Set model to evaluation mode (turns off dropout, etc.)

# 3. Generate Predictions
q_values_list = []
predictions = []

with torch.no_grad(): # Disable gradient tracking for testing 
    for sequence in X_test:
        state = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        
        # Get Q-values from the model
        q_vals = model(state)
        q_values_list.append(q_vals.cpu().numpy()[0])
        
        # Pick the action with the highest Q-value
        action = torch.argmax(q_vals).item()
        predictions.append(action)

q_values = np.array(q_values_list)
predictions = np.array(predictions)

# ==========================================
# TERMINAL OUTPUTS (TEXT)
# ==========================================

# Calculate overall accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ OVERALL MODEL ACCURACY: {accuracy * 100:.2f}%")

# Calculate Precision, Recall, and F1
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=['Normal (0)', 'Intrusion (1)']))

# Calculate specific security metrics
cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel() 

# Formulas for critical metrics
fpr = fp / (fp + tn)  # False Positive Rate
recall = tp / (tp + fn) # Detection Rate / True Positive Rate

print(f"\n🚨 CRITICAL IDS METRICS:")
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
plt.savefig('confusion_matrix.png', dpi=300)
print("\n📸 Saved 'confusion_matrix.png' to your folder.")
plt.show()

# 2. Generate and Save the ROC Curve
# Use the Q-value of the "Attack" action (index 1) as the confidence score
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
plt.savefig('roc_curve.png', dpi=300)
print("📸 Saved 'roc_curve.png' to your folder.")
plt.show()