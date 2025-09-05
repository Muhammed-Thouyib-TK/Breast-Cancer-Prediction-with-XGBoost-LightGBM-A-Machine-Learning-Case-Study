import os
import joblib
import streamlit as st
import gdown
import numpy as np

# Google Drive file ID 
FILE_ID = "1G1N70bS03ZLOhOmYK2wCrpT7lTXW9ZKz"
MODEL_PATH = "Breast_Cancer.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

def predictions(data):
    pred = model.predict(data)
    if pred[0] == 1:
        return 'The tumor is Malignant (Cancerous)'
    else:
        return 'The tumor is Benign (Non-cancerous)'

def main():
    # Title of the web app
    st.title('Breast Cancer Prediction Web App')

   # Feature names
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se',
        'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    # Features that should NOT be restricted to 0–1
    unbounded_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst'
    ]
    
    # Collect user input
    user_input = []
    for feature in feature_names:
        if feature in unbounded_features:
            # No 0–1 restriction
            value = st.number_input(f"{feature}: ", value=0.00000, step=0.00001, format="%.5f")
        else:
            # Restrict values between 0 and 1
            value = st.number_input(f"{feature}: ", min_value=0.0, max_value=1.0, value=0.0, step=None, format="%.5f")
        user_input.append(value)


    # Predict button
    if st.button('Predict'):
        data = np.array([user_input])  # Convert list to 2D numpy array
        result = predictions(data)
        st.success(result)

if __name__ == '__main__':
    main()
