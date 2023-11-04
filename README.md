# Diabetes Prediction with Logistic Regression - Code Documentation

## Introduction
This document provides an overview of the Python code developed for the Diabetes Prediction project using Logistic Regression. The code implements a machine learning model for predicting diabetes based on patient attributes such as pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age.

## Code Structure
The code consists of the following components:

### 1. Libraries Used
- numpy: For numerical operations and array manipulations.
- pandas: For data manipulation and analysis.
- pickle: For serializing and deserializing Python objects.
- FastAPI: A modern, fast web framework for building APIs.
- pydantic: For data validation and settings management.
- uvicorn: An ASGI server for running the FastAPI application.

### 2. Logistic Regression Implementation
#### sigmoid(x)
- A function that calculates the sigmoid (logistic) function, used for binary classification.

#### LogisticRegression class
- Implements logistic regression from scratch.
- __init__(self, lr=0.01, n_iters=1000): Constructor initializing learning rate and number of iterations.
- fit(self, X, y): Trains the logistic regression model using gradient descent.
- predict(self, X): Predicts the target variable based on the input features.

### 3. Loading Pretrained Model
- The code loads a pretrained logistic regression model from the finalized_model.sav file using pickle.

### 4. FastAPI Application
#### Data Input Model
- DiabetesInput(BaseModel): Pydantic model representing input data for diabetes prediction.

#### FastAPI Routes
- GET("/"): Serves the HTML content of the user interface from the index.html file.
- POST("/submit"): Accepts input data, processes it, and returns the predicted diabetes classification.

### 5. Application Execution
- The FastAPI application is run using the uvicorn.run() method, making the API accessible at http://0.0.0.0:8000.

## Usage
1. Ensure all required libraries are installed in your Python environment.
2. Run the script, which starts the FastAPI server.
3. Access the web interface at http://localhost:8000 to input patient data and receive diabetes predictions.

## Notes
- The logistic regression model is pretrained and loaded into memory using pickle.
- Input data for predictions is submitted through the /submit endpoint using a POST request.
- Predictions are made based on the input features and returned as binary classifications (0 or 1).

## Conclusion
This code implementation demonstrates the use of logistic regression for diabetes prediction and provides a user-friendly interface for end-users to interact with the model. It combines fundamental machine learning concepts with modern web development practices, creating a functional and accessible application for healthcare predictions.
