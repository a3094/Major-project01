import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
    precision_recall_curve
)

# === Load model ===
MODEL_PATH = os.path.join("models", "risk_model.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
classes = artifact["classes"]
soil_to_index = artifact["soil_to_index"]

# === Create or load test dataset ===
# Example simulated test data:
n_samples = 300
np.random.seed(42)
rainfall = np.random.uniform(50, 500, n_samples)
slope = np.random.uniform(0, 60, n_samples)
vegetation = np.random.uniform(0.1, 0.9, n_samples)
soil_types = np.random.choice(list(soil_to_index.keys()), n_samples)

# Encode soil
soil_idx = [soil_to_index[s] for s in soil_types]

# Features
X_test = np.column_stack((rainfall, slope, vegetation, soil_idx))

# True labels (simulate for demo; replace with real data)
y_true = np.random.choice([0, 1, 2], n_samples)  # 3 classes (Low, Medium, High risk)
y_pred = model.predict(X_test)

# === Metrics ===
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# === Confusion Matrix Heatmap ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === ROC Curve (for binary or one-vs-rest) ===
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)
    if y_score.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
    else:
        # One-vs-Rest ROC for multi-class
        for i in range(y_score.shape[1]):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curve")
        plt.legend()
        plt.show()

# === Precision-Recall Curve ===
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)
    if y_score.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
        plt.figure()
        plt.plot(recall, precision, color='purple')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()
