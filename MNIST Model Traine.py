import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def train_and_save_model():
    print("--- Training MNIST CNN Model ---")
    
    # 1. Load and Preprocess Data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and Reshape
    X_train = X_train.reshape((-1, 28, 28, 1)) / 255.0
    X_test = X_test.reshape((-1, 28, 28, 1)) / 255.0
    
    print("Data loaded and preprocessed.")

    # 2. Build CNN Architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # 3. Compile and Train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting Training (this may take a minute)...")
    model.fit(X_train, y_train, epochs=5, validation_split=0.1)

    # 4. Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # 5. Save the model
    model.save('mnist_cnn.keras')
    print("\n--- Model saved successfully as 'mnist_cnn.keras' ---")
    print("You can now run the Streamlit app.")

if __name__ == "__main__":
    train_and_save_model()
