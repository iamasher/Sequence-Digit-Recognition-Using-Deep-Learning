import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import os

def load_and_preprocess_data():
    """Load and preprocess MNIST data with enhanced preprocessing"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize and reshape
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test

def create_enhanced_model():
    """Create a more sophisticated CNN model with 3 conv blocks for higher accuracy"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third convolutional block for higher accuracy
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Custom optimizer with learning rate scheduling
    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_enhanced_model(model, X_train, y_train, X_test, y_test):
    """Train model with enhanced techniques"""
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)
    datagen.fit(X_train)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                                     save_best_only=True, mode='max')

    # Train with data augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                        epochs=10,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr, model_checkpoint])

    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Final evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    return history, test_acc

def save_plots_and_metrics(history, model, X_test, y_test):
    """Save all plots and metrics for later analysis"""
    # Create directory for results
    os.makedirs('training_results', exist_ok=True)

    # 1. Save training history plots
    plt.figure(figsize=(16, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results/training_metrics.png')
    plt.close()

    # 2. Save confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('training_results/confusion_matrix.png')
    plt.close()

    # 3. Save metrics summary
    with open('training_results/metrics_summary.txt', 'w') as f:
        f.write("=== Training Metrics Summary ===\n")
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final Test Accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")

def load_and_print_history():
    """Load and print saved training history"""
    with open('training_history.pkl', 'rb') as f:
        history = pickle.load(f)

    print("\n=== Saved Training History ===")
    print(f"Final Training Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Create enhanced model
    model = create_enhanced_model()
    model.summary()

    # Train with enhanced techniques
    history, test_acc = train_enhanced_model(model, X_train, y_train, X_test, y_test)

    # Save all plots and metrics
    save_plots_and_metrics(history, model, X_test, y_test)

    # Print saved history
    load_and_print_history()

    # Save the final model
    model.save('Trained_Model.h5')
    print("\nModel saved as 'Trained_Model.h5'")
    print("All training results saved in 'training_results' directory")

    return model

if __name__ == "__main__":
    model = main()