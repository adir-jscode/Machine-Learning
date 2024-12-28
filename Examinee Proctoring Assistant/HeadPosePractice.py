# Load the trained CNN model
#model = load_model("c:/Users/User/Desktop/Examinee Proctoring Assistant/Models/CNN_Model_1.keras")


import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("c:/Users/User/Desktop/Examinee Proctoring Assistant/Models/CNN_Model_1.keras")

# Class names (Update these according to your dataset)
class_names = ['Head_Movement_Allowed', 'Head_Movement_Not_Allowed']  # Change as per your dataset

def preprocess_image(image):
    """Preprocess the captured image to match the model input."""
    # Convert to grayscale as the model expects 1-channel input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size (width: 51, height: 60)
    image = cv2.resize(image, (51, 60))
    
    # Rescale pixel values (0-255 to 0-1)
    image = image.astype('float32') / 255.0
    
    # Add channel dimension (since the model expects shape (60, 51, 1))
    image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension (since the model expects a batch of images)
    image = np.expand_dims(image, axis=0)
    
    return image
    
    
    


# Capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Open webcam
    ret, frame = cap.read()  # Capture a single frame
    cap.release()  # Release the webcam
    if ret:
        return frame
    else:
        raise Exception("Failed to capture image from webcam.")

# Predict the head position
def predict_head_position(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    confidence = np.max(predictions[0])
    
    predicted_class = class_names[np.argmax(predictions[0])]
   
       
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Main execution
if __name__ == "__main__":
    # Capture image from webcam
    captured_image = capture_image()

     # Predict the head position
    predicted_class, confidence = predict_head_position(captured_image)

    # Print the results
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}%")
    
    # Show the captured image
    plt.imshow(cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY), 'gray')
    plt.axis('off')
    plt.title("Captured Image")
    plt.show()

   
