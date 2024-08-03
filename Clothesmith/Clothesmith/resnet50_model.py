import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image

# Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Build the model by running a dummy input
dummy_input = tf.random.normal([1, 224, 224, 3])
model(dummy_input)

# Load the precomputed embeddings and filenames

embeddings_path = r'D:\BCA Material\Research\AI ML Trainging\Clothsmith Project\Clothsmiths_Project\Clothesmith\Clothesmith\embeddings.pkl'
filenames_path = r'D:\BCA Material\Research\AI ML Trainging\Clothsmith Project\Clothsmiths_Project\Clothesmith\Clothesmith\filenames.pkl'
    

with open(embeddings_path, 'rb') as f:
    feature_list = pickle.load(f)

with open(filenames_path, 'rb') as f:
    image_paths = pickle.load(f)

# Fit the NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to find similar images
def find_similar_images(uploaded_img_path):
    uploaded_img_features = extract_features(uploaded_img_path, model)
    distances, indices = neighbors.kneighbors([uploaded_img_features], n_neighbors=6)  # +1 to exclude the query image itself

    local_image_paths = [path.replace('/content/updated_images/', 'D:\\BCA Material\\Research\\AI ML Trainging\\FashionRecm_2\\images\\images\\') for path in image_paths]
    similar_images = [(local_image_paths[i], distances[0][j]) for j, i in enumerate(indices[0]) if distances[0][j] > 0]  # Filter out the query image itself if included

    # Return paths of similar images
    return [img_path for img_path, _ in similar_images]
