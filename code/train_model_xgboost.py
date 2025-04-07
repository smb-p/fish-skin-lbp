import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import os

# Load features and labels
features_dir = "D:/research paper/results/features"

X_train = np.load(os.path.join(features_dir, "train_features.npy"))
y_train = np.load(os.path.join(features_dir, "train_labels.npy"))
X_test = np.load(os.path.join(features_dir, "test_features.npy"))
y_test = np.load(os.path.join(features_dir, "test_labels.npy"))

# Train XGBoost classifier
print("Training XGBoost model...")
clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
clf.fit(X_train, y_train)
print("Training complete!")

# Evaluate on test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[
    "Argulus",
    "Broken antennae and rostrum",
    "EUS",
    "Red Spot",
    "Tail And Fin Rot",
    "THE BACTERIAL GILL ROT"
]))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "Argulus",
    "Broken antennae and rostrum",
    "EUS",
    "Red Spot",
    "Tail And Fin Rot",
    "THE BACTERIAL GILL ROT"
])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Save the trained model
predictions_dir = "D:/research paper/results/predictions"
os.makedirs(predictions_dir, exist_ok=True)

model_path = os.path.join(predictions_dir, "fish_disease_model_xgboost.pkl")
joblib.dump(clf, model_path)
print(f"\nModel saved to {model_path}")