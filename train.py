# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set Directories
train_dir = './dataset/train'
test_dir = './dataset/test'

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Preprocess grayscale to 3 channels
def preprocess_image(image):
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,) * 3, axis=-1)  # Duplicate channels
    return image

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # unless you're sure flipping is safe
    fill_mode='nearest',
    brightness_range=[0.9, 1.1],
    preprocessing_function=preprocess_image
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255,
    preprocessing_function=preprocess_image
)

train_datagen.preprocessing_function = preprocess_image # Apply preprocess_image
test_datagen.preprocessing_function = preprocess_image # Apply preprocess_image

# Load Images and Apply Preprocessing
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False  # Keep order consistent for evaluation
)

# Display Class Imbalance
class_counts = train_generator.classes
class_labels = list(train_generator.class_indices.keys())
print(f"Class Distribution: {np.bincount(class_counts)}")

# Compute Class Weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_counts),
    y=class_counts
)
class_weights = dict(enumerate(class_weights))
print(f"Computed Class Weights: {class_weights}")

def get_file_paths_and_labels(directory):
    filepaths = []
    labels = []
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_dir, filename))
                labels.append(class_name)
    
    return filepaths, labels, class_names

train_filepaths, train_labels, class_names = get_file_paths_and_labels(train_dir)

label_to_index = {name: index for index, name in enumerate(class_names)}

train_label_indices = np.array([label_to_index[label] for label in train_labels])

train_df = pd.DataFrame({'filepaths': train_filepaths, 'labels': train_labels})

def visualize_augmented_images(generator, class_names, num_images=6):
    x_batch, y_batch = next(generator)
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(x_batch[i])
        label_index = np.argmax(y_batch[i])
        plt.title(f"Label: {class_names[label_index]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_augmented_images(train_generator, class_names)

# Create VGG16 Model with Fine-Tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the initial layers (first 5 layers)
for layer in base_model.layers[:5]:
    layer.trainable = False

# Unfreeze the last few layers for fine-tuning (from layer 5 onwards)
for layer in base_model.layers[5:]:
    layer.trainable = True

# Define Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler, model_checkpoint]
)

# Visualize Training Results
def plot_results(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_results(history)

# Evaluate Model
test_generator.reset()
y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
accuracy = np.mean(y_pred_classes == y_true)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# prompt: save model dengan nama tgl dan jam saat ini

from datetime import datetime

# Get current date and time
now = datetime.now()

# Format date and time as a string
timestamp = now.strftime("%Y%m%d_%H%M%S")

# Define the save path with the timestamp
save_path = f'model_{timestamp}.keras'

# Save the model
model.save(save_path)

print(f"Model saved to {save_path}")

# Load the best saved model
# loaded_model = tf.keras.models.load_model('model_20250603_150458.keras')
loaded_model = tf.keras.models.load_model(save_path)

# You can now use 'loaded_model' for evaluation or prediction
# For example, you can evaluate it on the test set:
loss, accuracy = loaded_model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
print(f"Loaded Model - Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")

# Or make predictions:
# predictions = loaded_model.predict(new_data_generator)

# Get file paths and labels for the training data
train_filepaths, train_labels, class_names = get_file_paths_and_labels(train_dir)

# Convert labels to numerical format
label_to_index = {name: index for index, name in enumerate(class_names)}
train_label_indices = np.array([label_to_index[label] for label in train_labels])

# Store cross-validation results
fold_accuracies = []
# fold_histories = []  # Used only during train

# Perform K-Fold Cross-Validation (evaluation only)
for fold, (train_index, val_index) in enumerate(kf.split(train_filepaths, train_label_indices)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")

    # Split data for the current fold
    train_fold_filepaths = [train_filepaths[i] for i in train_index]
    train_fold_labels = [train_labels[i] for i in train_index]
    val_fold_filepaths = [train_filepaths[i] for i in val_index]
    val_fold_labels = [train_labels[i] for i in val_index]

    # Create DataFrame for generators
    # train_fold_df = pd.DataFrame({'filepaths': train_fold_filepaths, 'labels': train_fold_labels})  # Only need if you want to practice again
    val_fold_df = pd.DataFrame({'filepaths': val_fold_filepaths, 'labels': val_fold_labels})

    # Create validation data generator
    val_fold_generator = test_datagen.flow_from_dataframe(
        val_fold_df,
        x_col='filepaths',
        y_col='labels',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False  # Keep order consistent for evaluation
    )

    # Evaluate the model on the validation set of the current fold
    val_fold_generator.reset()
    y_pred_fold = loaded_model.predict(val_fold_generator, steps=val_fold_generator.samples // val_fold_generator.batch_size + 1)
    y_pred_classes_fold = np.argmax(y_pred_fold, axis=1)

    # Get true labels for the validation set of the current fold
    y_true_fold = val_fold_generator.classes

    fold_accuracy = accuracy_score(y_true_fold, y_pred_classes_fold)
    print(f"Fold {fold+1} Accuracy: {fold_accuracy * 100:.2f}%")
    print(classification_report(y_true_fold, y_pred_classes_fold, target_names=class_labels))

    fold_accuracies.append(fold_accuracy)

# Print average accuracy across all folds
print("\n--- Cross-Validation Results ---")
print(f"Average Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies) * 100:.2f}%")