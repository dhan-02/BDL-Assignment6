# BDL-Assignment6
Assignment 6 for the Big Data Lab course - Jan - May 2024. FastAPI for digit classification. 

## Instructions on how to run
The repository consists of two script files task1.py and task2.py. The test_images folder contains sample images use din task1 and task2. The bin folder contains the trained keras model. The dependencies are all stored in requirements.txt. <br />
Use conda create --name my_env --file requirements.txt to create an environment with required libraries. <br />
Use python unittests.py to run the unit tests <br />
Use python task2.py bin/mnist-digitclassifier-model.keras to run the app <br />
Note that all of above commands need to be run from the root folder of the repo. <br />

## Usage
Use the app through the Swagger UI provided by FastApi <br />
i.e if the app is being run on server 127.0.0.1 and port 5000 go to 127.0.0.1:5000/docs <br />
Note that the above functionality (Swagger UI) works smoothly on Firefox but faces some issues on Chrome <br />
Upload the image in the given placeholder and the predicted digit can be viewed <br />
![Screenshot from 2024-04-27 22-53-20](https://github.com/dhan-02/BDL-Assignment6/assets/74642765/10b99121-80af-4156-affa-022a9cc7562c)

## Problem Statement
#### Task 1: 
1. Create a FastAPI module. <br />
2. Take the path of the model as a command line argument. <br />
3. Create a function “def load_model(path:str) -> Sequential” which will load the model saved at the
supply path on the disk and return the keras.src.engine.sequential.Sequential model. <br />
4. Create a function “def predict_digit(model:Sequential, data_point:list) -> str” that will take the
image serialized as an array of 784 elements and returns the predicted digit as string. <br />
5. Create an API endpoint “@app post(‘/predict’)” that will read the bytes from the uploaded image
to create an serialized array of 784 elements. <br />
6. Test the API via the Swagger UI (<api endpoint>/docs) or Postman, where you will upload the digit
as an image (28x28 size). <br />
#### Task 2: 
1. Create a new function “def format_image” which will resize any uploaded images to a 28x28 grey scale image followed by creating a serialized array of 784 elements. <br />
2. Modify Task 1 to incorporate “format_image” inside the “/predict” endpoint to preprocess any uploaded content. <br />
3. Upload hand drawn images to your API and find out if your API is able to figure out the digit correctly. Repeat this exercise for 10 such drawings and report the
performance of your API/model combo. <br />

## Performance
Accuracy on handdrawn images (Task 2) : 50 %
<br />
<br />
![Screenshot from 2024-04-27 23-22-40](https://github.com/dhan-02/BDL-Assignment6/assets/74642765/bdf185b5-2515-404b-8126-8c28390825cf)









