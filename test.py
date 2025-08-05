import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator
from sklearn.metrics import accuracy_score

# Load the saved model
model = load_model('cnn_model.keras')  # Replace with the path to your saved model

# Assuming you have a test generator or test dataset to evaluate
test_datagen = ImageDataGenerator(rescale=1./255)  # Scale the test data
test_generator = test_datagen.flow_from_directory(
    'test',  # Path to test data
    target_size=(224, 224),  # Resize images as per the model input
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f"Test Accuracy: {test_acc * 100:.2f}%")
