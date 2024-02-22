# Age-Gender-Prediction-
Implemented Age and Gender Prediction using VGG16, data augmentation, multi-output model, Streamlit app. #DeepLearning #ComputerVision

This repository contains a Deep Learning model for predicting age and gender from facial images. The model is built using the VGG16 architecture, trained on the UTKFace dataset. It includes data augmentation techniques, a multi-output neural network, and a Streamlit web app for easy prediction.

Overview:

    Model Architecture: VGG16 with added dense layers for age and gender prediction.
    Dataset: UTKFace dataset for training and testing.
    Data Augmentation: Used to improve model generalization.
    Loss Functions: Mean Absolute Error for age, Binary Crossentropy for gender.
    Metrics: MAE for age, Accuracy for gender.
    Streamlit App: Provides an interactive interface for uploading images and getting predictions.

File Structure:

    age_gender_prediction.py: Python script containing the model implementation, training, and Streamlit app.
    main.py: Python script for testing the model in the terminal.
    UTKFace/: Folder containing the UTKFace dataset (not included in the repository due to size).

Usage:

    Setup Environment:
        Install required libraries

    Training:
        Run the script age_gender_prediction.py to train the model. Adjust parameters as needed.
        Run main.py in the terminal.

    Prediction:
        After training, the Streamlit app can be launched with streamlit run age_gender_prediction.py.
        Upload an image to get predictions for age and gender.

Notes:

    The model was trained on a subset of the UTKFace dataset (20,000 images) for demonstration purposes.
    Feel free to experiment with different architectures, hyperparameters, and datasets.
    Suggestions for improving efficiency include:
        Loading the pre-trained model once at app startup.
        Implementing batch processing for uploaded images.
        Using GPU acceleration for faster computations.

References:

    UTKFace Dataset: https://susanqq.github.io/UTKFace/
    VGG16: https://keras.io/api/applications/vgg/
    Streamlit Documentation: https://docs.streamlit.io/

Authon: Shine Gupta
