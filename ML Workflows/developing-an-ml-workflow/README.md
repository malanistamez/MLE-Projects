# Project: ML Workflow for Scones Unlimited on Amazon SageMaker

## Overview
Scones Unlimited aims to leverage machine learning to optimize their production processes and enhance customer satisfaction. In this project, we build and deploy a machine learning model on Amazon SageMaker to classify images of scones. The workflow encompasses data preparation, model training, deployment, API construction, and monitoring.

## Setup

### SageMaker Studio Workspace
Ensure the setup of a SageMaker Studio workspace with an appropriate kernel to execute the project. Refer to the provided documentation for detailed instructions on setting up SageMaker Studio.

### Data Loading and Preparation
Complete the ETL (Extract, Transform, Load) process to prepare the dataset for machine learning with SageMaker. The dataset comprises images of various types of scones.

## Model Training

### Training the Machine Learning Model
Follow the instructions in the provided starter code to train the machine learning model. Ensure completion of the training process up to the "Getting ready to deploy" section, indicating successful model training for image classification.

### Constructing the API Endpoint
Deploy the trained ML model on SageMaker to construct an API endpoint. Ensure that the unique model endpoint name is printed in the notebook for future reference.

### Sample Image Predictions
Make predictions using sample images to validate the deployed model's performance.

## Machine Learning Workflow

### Authoring Lambda Functions
Develop three Lambda functions responsible for different aspects of the machine learning workflow:
1. **Image Data Retrieval (`serialize_image_data.py`)**: Retrieve image data and pass it to the Step Function.
2. **Image Classification (`classify_image.py`)**: Perform image classification using the trained model.
3. **Inference Filtering (`filter_inference.py`)**: Filter low-confidence inferences from the classification results.

Save the code for each Lambda function in a Python script as per the project specifications.

### Authoring Step Function
Compose the Lambda functions together in a Step Function to orchestrate the machine learning workflow. Export the Step Function definition as a JSON file (`step-functions.json`) and capture a screenshot of the working Step Function.

## Model Monitoring

### Extracting Monitoring Data
Extract monitoring data from S3 to analyze the model's performance and identify any errors or anomalies.

### Visualizing Model Monitor Data
Load and visualize the Model Monitor data in the notebook to gain insights into the model's behavior and performance.

## Project Files

- **lambdas directory**:
  - `classify_image.py`: Lambda function for image classification.
  - `filter_inference.py`: Lambda function for filtering inferences.
  - `serialize_image_data.py`: Lambda function for serializing image data.

- **starter.ipynb**: Jupyter notebook containing the project's starter code and instructions.

- **step-functions.json**: JSON export of the Step Function definition.

## Screenshots

- **Step function Successful**:
  ![Lambda Successful](step_function_success.tiff)

- **Step function Threshold Not Met**:
  ![Threshold Not Met](step_func_threshold_not_success.tiff)

By following these steps and utilizing the provided files, we ensure the successful construction and deployment of a robust machine learning workflow for Scones Unlimited on Amazon SageMaker, facilitating optimized production processes and enhanced customer satisfaction.