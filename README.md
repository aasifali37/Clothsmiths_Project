# Fashion Recommendation Model

## Overview
This project implements a fashion recommendation system using a ResNet50 model. The system is designed to provide recommendations based on image similarity. This README file provides instructions on how to set up and run the project.

## File Structure
- **app.py**: The main application script.
- **butler-icon.png**: An icon used in the application.
- **data.csv**: The dataset file containing image data.
- **dataset_download.txt**: Instructions for downloading the dataset.
- **resnet50_model.py**: Script defining the ResNet50 model used for feature extraction.
- **Resnet Model pickles files.txt**: List of necessary pickle files.
- **Cloth-Smith.docx**: Project report.
- **step_by_step_installation_Dataset**: Step-by-step instructions for installing the dataset.

## Prerequisites
- Python 3.6 or higher
- Required Python libraries (listed in `requirements.txt`)

## Project Workflow

### Data Preprocessing
1. **Load Data**: Read the `data.csv` file which contains image data.
2. **Preprocess Data**: Clean and preprocess the data as required.

### Feature Extraction
1. **Load Pretrained Model**: Define and load the pretrained ResNet50 model with GlobalMaxPooling.
2. **Load Images**: Load images from the specified folder.
3. **Extract Features**: Use the model to extract features from images.
4. **Save Features**: Save extracted features and filenames into pickle files for later use.

### Recommendation System
1. **Initialize Vectorizer**: Initialize the vectorizer for processing text data.
2. **Train Model**: Train the model using TF-IDF.
3. **Input Query**: Accept user input query.
4. **Vectorize Query**: Convert the input query into a vector.
5. **Calculate Similarity**: Compute cosine similarity between the query vector and image feature vectors.
6. **Sort Results**: Sort the results based on similarity.
7. **Display Recommendations**: Display the top recommended items.

## Additional Resources
- **Cloth-Smith.docx**: Detailed project report and documentation.
- **step_by_step_installation_Dataset**: Detailed instructions for dataset installation.

For any questions or issues, please contact [kritbarnwal5004@gmail.com,aasifali0203@gmail.com].

