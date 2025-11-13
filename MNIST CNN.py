import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def run_mnist_project():
    print("--- Task 2: MNIST Digit Classification (CNN) ---")

    # 1. Load and Preprocess Data
    # MNIST contains 60k training images and 10k test images of handwritten digits (0-9)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Reshape data to include channel dimension (28, 28, 1) for Grayscale
    # CNNs expect shape (batch, height, width, channels)
    X_train = X_train.reshape((-1, 28, 28, 1))
    X_test = X_test.reshape((-1, 28, 28, 1))

    print(f"Training data shape: {X_train.shape}")

    # 2. Build CNN Architecture
    model = models.Sequential([
        # Convolutional Layer: Extracts features using 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Pooling Layer: Reduces spatial dimensions
        layers.MaxPooling2D((2, 2)),
        # Second Conv Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Flatten: Converts 2D matrix to 1D vector
        layers.Flatten(),
        # Dense Layer: Fully connected classification
        layers.Dense(64, activation='relu'),
        # Output Layer: 10 neurons (one for each digit 0-9), softmax for probability
        layers.Dense(10, activation='softmax')
    ])

    model.summary()

    # 3. Compile and Train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nStarting Training (this may take a minute)...")
    # Training for 5 epochs is usually enough to reach >98% on MNIST
    history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

    # 4. Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # 5. Visualization
    plot_predictions(model, X_test, y_test)

def plot_predictions(model, X_test, y_test):
    # Get predictions for the first 5 images
    predictions = model.predict(X_test[:5])
    
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        # Reshape back to 28x28 for display
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        pred_label = np.argmax(predictions[i])
        true_label = y_test[i]
        
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    print("\nDisplaying prediction plot...")
    plt.show()

if __name__ == "__main__":
    run_mnist_project()
