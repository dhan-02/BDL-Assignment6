import unittest
import sys
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
from task2 import app, preprocess_image, predict_digit,load_my_model,format_image

class TestApp(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.model_path = "bin/mnist-digitclassifier-model.keras"  # Provide the path to your model here

    def test_predict_digit(self):
        # Sample data for testing
        sample_data = [0.5] * 784  # Assuming 784 elements for a grayscale image

        # Load model for testing
        loaded_model = load_my_model(self.model_path)

        # Test predict_digit function
        predicted_digit = predict_digit(loaded_model, sample_data)
        self.assertIsInstance(predicted_digit, str)
        self.assertRegex(predicted_digit, r'^[0-9]$')  # Predicted digit should be a single digit

    def test_preprocess_image(self):
        # Load a sample image for testing
        with open("test_images/task1/sample_0.png", "rb") as image_file:
            sample_image = image_file.read()

        # Test preprocess_image function
        processed_image = preprocess_image(sample_image)
        self.assertIsInstance(processed_image, list)
        self.assertEqual(len(processed_image), 784)  # Serialized array should have 784 elements

        # Test with different image formats
        with open("test_images/task1/sample_0.png", "rb") as image_file:
            sample_image = image_file.read()
        processed_image = preprocess_image(sample_image)
        self.assertEqual(processed_image, processed_image)  # Both images should preprocess the same

        # Test with invalid image data
        with self.assertRaises(ValueError):
            preprocess_image(None)  # Should raise ValueError for None input
    
    def test_format_image(self):
        # Test resizing an image to 28x28
        sample_image = Image.open("test_images/task1/sample_0.png")
        formatted_img = format_image(sample_image)
        self.assertEqual(formatted_img.size, (28, 28))
    
    def test_load_my_model_success(self):
        # Test loading a valid model
        loaded_model = load_my_model(self.model_path)
        self.assertIsNotNone(loaded_model)

    def test_load_my_model_failure(self):
        # Test loading a non-existent model
        with self.assertRaises(ValueError):
            load_my_model("non_existent_model.h5")

if __name__ == '__main__':
    # Check if the path to the model is provided as a command line argument
    unittest.main()
