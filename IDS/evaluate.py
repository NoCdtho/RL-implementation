import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataSet import X_test, y_test
from tensorflow.keras.models import load_model

print("Evaluating LSTM-DQN on Test Dataset...")

main_network = load_model("lstm_dqn_model.keras")

# 1. Ask the trained network to predict the test dataset
# Keras predict handles the batching automatically
q_values = main_network.predict(X_test, batch_size=32)

# Pick the action with the highest Q-value for each step
predictions = np.argmax(q_values, axis=1)

# 2. Calculate Precision, Recall, F1, and Accuracy
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=['Normal (0)', 'Intrusion (1)']))

# 3. Calculate False Positive Rate and exact Recall
cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel() # True Negative, False Positive, False Negative, True Positive

fpr = fp / (fp + tn)
recall = tp / (tp + fn)

print(f"\nCRITICAL IDS METRICS:")
print(f"Recall (Detection Rate): {recall:.4f} (Goal: As close to 1.0 as possible)")
print(f"False Positive Rate: {fpr:.4f} (Goal: As close to 0.0 as possible)")

# 4. Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Intrusion'], 
            yticklabels=['Normal', 'Intrusion'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('LSTM-DQN Confusion Matrix')
plt.show()