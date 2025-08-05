import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

# Load XGBoost model
xgb_model = XGBRegressor()
xgb_model.load_model("xgboost_model.json")

# Load CNN model
cnn_model = load_model("cnn_model.keras")

# Define class labels based on the dataset
class_labels = [
    "Early Blight", "Healthy", "Late Blight", "Leaf Miner",
    "Magnesium Deficiency", "Nitrogen Deficiency",
    "Potassium Deficiency", "Spotted Wilt Virus"
]

# Define yield impact values for each class
impact_values = [90, 100, 70, 50, 60, 40, 30, 20]  # Adjust impacts as needed

def map_cnn_to_yield(probabilities, impact_values):
    """
    Map CNN output probabilities to a single yield percentage.
    probabilities: list of class probabilities
    impact_values: list of numerical impacts corresponding to each class
    """
    return np.dot(probabilities, impact_values)  # Weighted sum of probabilities and impacts

def predict_integrated_yield_with_severity(xgboost_input, cnn_image, weights=(0.6, 0.4)):
    """
    Combine XGBoost and CNN predictions using weighted averaging
    and detect the severity class based on the CNN output.
    xgboost_input: Weather data input for XGBoost (scaled)
    cnn_image: Preprocessed image input for CNN
    weights: Tuple indicating weights for XGBoost and CNN outputs
    """
    try:
        # Predict yield percentage from XGBoost
        xgb_output = xgb_model.predict(xgboost_input)
        
        # Predict class probabilities from CNN
        cnn_output = cnn_model.predict(cnn_image)
        cnn_probabilities = cnn_output[0]  # Assuming single image input
        
        # Detect the class with the highest probability
        predicted_class_index = np.argmax(cnn_probabilities)
        detected_class = class_labels[predicted_class_index]
        detected_probability = cnn_probabilities[predicted_class_index]
        
        # Map CNN probabilities to yield value
        cnn_yield = map_cnn_to_yield(cnn_probabilities, impact_values)
        
        # Combine predictions using weighted averaging
        integrated_yield = (weights[0] * xgb_output[0]) + (weights[1] * cnn_yield)
        
        # Print results
        print(f"Detected Disease: {detected_class} (Probability: {detected_probability:.2f})")
        return integrated_yield, detected_class, detected_probability
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

# Path to the image folder
image_folder = r"C:\Users\kisha\Downloads\Tomato-Village-main\Tomato-Village-main\Variant-c(Object Detection)\train\images"

# Function to load and preprocess a random image
def load_random_image(folder_path, target_size=(224, 224)):
    """
    Randomly selects an image from the folder, loads, and preprocesses it.
    """
    # Get list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly select an image
    random_image_file = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image_file)
    
    # Load and preprocess the image
    image = load_img(image_path, target_size=target_size)  # Resize the image
    image_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    print(f"Using image: {random_image_file}")
    return image_array

# Example XGBoost input 
xgboost_input = np.array([[0.52, 0.12, 0.46, 0.95, 0.60, 0.20]])  # Scaled input features

# Load a random image and predict yield with severity detection
cnn_image = load_random_image(image_folder)
predicted_yield, detected_class, detected_probability = predict_integrated_yield_with_severity(xgboost_input, cnn_image)

if predicted_yield is not None:
    print(f"Predicted Crop Yield Percentage: {predicted_yield:.2f}%")