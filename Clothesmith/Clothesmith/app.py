import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from resnet50_model import find_similar_images
import matplotlib.image as mimg

# Set the page title and icon
st.set_page_config(page_title="Clothsmiths", page_icon="butler-icon.png")

# Load data
data = pd.read_csv(r"D:\BCA Material\Research\AI ML Trainging\Clothsmith Project\Clothsmiths_Project\Clothesmith\Clothesmith\data.csv")

# Base folder path for images
image_folder = r"D:\BCA Material\Research\AI ML Trainging\Clothsmith Project\images\images"

# Initialize session state variables
if 'selected_image_path' not in st.session_state:
    st.session_state.selected_image_path = None

if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = data.copy()

# Function to get unique options based on filter
def get_unique_options(df, column, filter_column=None, filter_value=None):
    if filter_column and filter_value:
        df = df[df[filter_column] == filter_value]
    return df[column].unique()

# Sidebar for filters
st.sidebar.title("Clothsmiths 👓 Search")

# Select Gender
gender = st.sidebar.selectbox("Select Gender:", ['Select'] + list(get_unique_options(data, 'gender')))
if gender != 'Select':
    # Filter by gender
    filtered_data = data[data['gender'] == gender]

    # Select Usage
    usage_options = ['Select'] + list(get_unique_options(filtered_data, 'usage'))
    usage = st.sidebar.selectbox("Select Usage:", usage_options)
    
    if usage != 'Select':
        # Filter by usage
        filtered_data = filtered_data[filtered_data['usage'] == usage]

        # Select Article Type
        article_type_options = ['Select'] + list(get_unique_options(filtered_data, 'articleType'))
        article_type = st.sidebar.selectbox("Select Article Type:", article_type_options)
        
        if article_type != 'Select':
            # Filter by article type
            st.session_state.filtered_df = filtered_data[filtered_data['articleType'] == article_type]
        else:
            st.session_state.filtered_df = filtered_data
    else:
        st.session_state.filtered_df = filtered_data
else:
    st.session_state.filtered_df = data

# Text input for additional query
query = st.text_input("Enter your search query:", "")

# Preprocess function for text
def preprocess(text):
    return text.lower()

# Apply preprocessing
st.session_state.filtered_df.loc[:, 'processed_display_name'] = st.session_state.filtered_df['productDisplayName'].dropna().apply(preprocess)

# Vectorization and similarity calculation
if query:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(st.session_state.filtered_df['processed_display_name'])
    query_vec = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    st.session_state.filtered_df.loc[:, 'similarity'] = similarity
    st.session_state.filtered_df = st.session_state.filtered_df.sort_values(by='similarity', ascending=False)

# Display filtered and sorted results
if st.session_state.filtered_df.empty:
    st.write("No results match your query.")
else:
    st.write(f"Showing results for: {query}")
    images_to_display = st.session_state.filtered_df.head(4)  # Limit to top 4 results

    # Create 2x2 grid
    cols = st.columns(2)
    for i, (_, row) in enumerate(images_to_display.iterrows()):
        image_filename = f"{row['id']}.jpg"
        image_path = os.path.join(image_folder, image_filename)
        if os.path.exists(image_path):
            with cols[i % 2]:
                st.image(image_path, width=150)
                if st.button('Select', key=row['id']):
                    st.session_state.selected_image_path = image_path

# Show selected image path
if st.session_state.selected_image_path:
    #st.write(f"Selected image path: {st.session_state.selected_image_path}")

    # Find similar images
    selected_image_id = os.path.splitext(os.path.basename(st.session_state.selected_image_path))[0]
    # Fetch the description for the selected image
    selected_image_description = data[data['id'] == int(selected_image_id)]['productDisplayName'].values[0]
    st.write(f"Selected image description:\n {selected_image_description}")
    similar_image_paths = find_similar_images(st.session_state.selected_image_path)

    # Display similar images
    if similar_image_paths:
        st.write("Similar images:")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(similar_image_paths):
                similar_image_path = similar_image_paths[idx]
                image = mimg.imread(similar_image_path)
                col.image(image, use_column_width=True)
                image_id = os.path.splitext(os.path.basename(similar_image_path))[0]
                # Fetch the description for this image
                image_description = data[data['id'] == int(image_id)]['productDisplayName'].values[0]
                col.write(f"Description:\n\n {image_description}")
                #col.write(f"Path: {similar_image_path}")

    # Add some creative styling
    st.markdown("""
        <style>
            .css-1aumxhk {
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

else:
    st.write("No image selected.")
