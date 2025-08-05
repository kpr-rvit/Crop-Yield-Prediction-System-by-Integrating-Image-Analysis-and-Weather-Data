import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Get current directory
base_dir = os.path.dirname(os.path.realpath(__file__))  # Folder where the script is located

# Paths to the datasets
train_dir = os.path.join(base_dir, 'train')  # Training data
val_dir = os.path.join(base_dir, 'val')  # Validation data
test_dir = os.path.join(base_dir, 'test')  # Test data

# Check if directories exist
assert os.path.exists(train_dir), f"Training directory {train_dir} does not exist."
assert os.path.exists(val_dir), f"Validation directory {val_dir} does not exist."
assert os.path.exists(test_dir), f"Test directory {test_dir} does not exist."

# Image data generators for data augmentation and loading images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  
test_datagen = ImageDataGenerator(rescale=1./255)  

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to training data
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical'  # Specifies multi-class classification (one-hot encoded labels)
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,  # Path to validation data
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,  # Path to test data
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical'
)

# Convert the training generator to a tf.data.Dataset 
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
    )
)

train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for performance optimization

# Build the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten the output from convolutional layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(train_generator.num_classes, activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model with dynamic steps calculation
history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Automatically calculated
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,  # Automatically calculated
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save the trained model
model.save('cnn_model.keras')  # Save the model in the recommended Keras Format 