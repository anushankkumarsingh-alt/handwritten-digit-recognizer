# Handwritten Digit Recognizer
# CNN using the MNIST dataset

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("🤖 Handwritten Digit Recognizer")
print("--------------------------------")

# 1. Load the MNIST dataset
print("\n📥 Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training images: {x_train.shape}")
print(f"Testing images: {x_test.shape}")

# 2. Normalize pixel values
# Convert values from 0-255 to 0-1

x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Add channel dimension for CNN
# Images become: 28 x 28 x 1

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 4. Build CNN model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

# 5. Show model architecture

print("\n🧠 CNN Model:")
model.summary()

# 6. Compile model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 7. Train model

print("\n🚀 Training model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

# 8. Evaluate model

print("\n📊 Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# 9. Make predictions for first 10 images

predictions = model.predict(
    x_test[:10],
    verbose=0
)

print("\n🔮 Predictions for first 10 test images:")

for i in range(10):
    predicted_digit = np.argmax(predictions[i])
    actual_digit = y_test[i]

    print(
        f"Image {i + 1}: "
        f"Predicted = {predicted_digit}, "
        f"Actual = {actual_digit}"
    )

# 10. Display first 10 predictions

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_test[i].squeeze(),
        cmap="gray"
    )

    predicted_digit = np.argmax(predictions[i])

    plt.title(
        f"Pred: {predicted_digit}\n"
        f"Actual: {y_test[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# 11. Create predictions for the entire test dataset

print("\n📊 Creating confusion matrix...")

all_predictions = model.predict(
    x_test,
    verbose=0
)

predicted_labels = np.argmax(
    all_predictions,
    axis=1
)

# 12. Create confusion matrix

cm = confusion_matrix(
    y_test,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

disp.plot(cmap="Blues")

plt.title(
    "MNIST Digit Recognition - Confusion Matrix"
)

plt.show()

# 13. Save trained model

model.save("mnist_digit_model.keras")

print("\n💾 Model saved as mnist_digit_model.keras")

print("\n🎉 Training and evaluation complete!")
