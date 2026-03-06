
# Evaluate the model on validation/test dataset
loss, accuracy = model.evaluate(valid_gen)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Generate Predictions

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(valid_gen)

y_pred = np.argmax(predictions, axis=1)
y_true = valid_gen.classes

class_labels = list(valid_gen.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save Final Model

model.save("final_plant_disease_model.h5")

print("\nFinal model saved successfully!")
