# Classification Project

This project focuses on building a classification model to classify natural disasters-related messages into different categories. The goal is to develop a model that can accurately identify the category of a given message, which can be helpful in various scenarios such as disaster response, customer support, or content moderation.

## Project Files

The project consists of the following files:

1. **app**
   - `run.py`: The main script to run the web application for message classification.
2. **data**
   - `messages.csv`: text data with the messages that will be used to train the model.
   - `categories`: series of binary variables that identify each text message.
   - `etl_pipeline.py`: Script to load, preprocess, clean, and save the messages data into the SQLite database.

3. **models**
   - `train_classifier.py`: Script to load the preprocessed data from the SQLite database, train a classification model, and save the trained model as a pickle file.

## Project Workflow

The project workflow involves the following steps:

1. **Data Processing**: The `process_data.py` script is used to load and preprocess the messages data. It performs tasks such as data cleaning, feature extraction, and storing the processed data into a SQLite database.

2. **Model Training**: The `train_classifier.py` script is responsible for loading the preprocessed data from the SQLite database, training a classification model using machine learning techniques, and saving the trained model as a pickle file.

3. **Web Application**: The `run.py` script runs a web application that allows users to input messages and receive classification results. It uses the trained model to classify the messages and displays the predicted categories.

To run the project, follow these steps:

1. Execute the `process_data.py` script to preprocess and store the messages data into the SQLite database.

2. Run the `train_classifier.py` script to train the classification model using the preprocessed data and save the trained model as a pickle file.

3. Finally, run the `run.py` script to start the web application and classify messages using the trained model.
