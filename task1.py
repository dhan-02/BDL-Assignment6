import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from fastapi import FastAPI, File, UploadFile
from typing import List
from PIL import Image
import numpy as np
from io import BytesIO
import uvicorn


app = FastAPI(docs_url="/docs")

# Function to load the model from the specified path
def load_my_model(path: str) -> Sequential:
    """
    Load the model saved at the supplied path on the disk.

    Args:
    path (str): The path to the saved model on disk.

    Returns:
    keras.src.engine.sequential.Sequential: The loaded Keras Sequential model.
    """
    loaded_model = load_model(path)
    return loaded_model

# Function to predict the digit using the loaded model
def predict_digit(model: Sequential, data_point: list) -> str:
    """
    Predict the digit using the loaded model.

    Args:
    model (keras.src.engine.sequential.Sequential): The loaded Keras Sequential model.
    data_point (list): The image serialized as an array of 784 elements.

    Returns:
    str: The predicted digit as a string.
    """
    # Preprocess the data_point (reshape and normalize)
    data_point = np.array(data_point).reshape(1, -1)
    
    # Predict the digit
    prediction = model.predict(data_point)
    
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    
    return str(predicted_digit)

def preprocess_image(file) -> List[float]:
    img = Image.open(BytesIO(file))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28*28)  # Flatten to 1D array
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array.tolist()[0]

@app.post("/predict")
async def predict_image_digit(file: UploadFile = File(...)):
    # Read the bytes from the uploaded image
    contents = await file.read()
    
    # Preprocess the image to create a serialized array of 784 elements
    processed_image = preprocess_image(contents)
    
    # Get the path to the model from the command line argument
    model_path = sys.argv[1]
    print(model_path)
    
    # Load the model
    loaded_model = load_my_model(model_path)

    # Predict the digit using the serialized array
    predicted_digit = predict_digit(loaded_model,processed_image)
    print("Hi")
    # Return the predicted digit to the client
    return {"digit": predicted_digit}

if __name__ == "__main__":
    # Check if the path to the model is provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_model>")
        sys.exit(1)
    
#     # # Example of how to use the predict_digit function
#     # example_data_point = [100] * 784 
#     # predicted_digit = predict_digit(loaded_model, example_data_point)
#     # print("Predicted digit:", predicted_digit)
    uvicorn.run(app, host="127.0.0.1", port=5000)
