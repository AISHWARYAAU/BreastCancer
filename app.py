import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the model file
breast_cancer_model_path = os.path.join(working_dir, 'BreastCancer.sav')

# Load the pre-trained model
with open(breast_cancer_model_path, 'rb') as model_file:
    BreastCancer_model = pickle.load(model_file)

# Values from the provided data
sample_values = {
    "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0, "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776, "concavity_mean": 0.3001, "concave_points_mean": 0.1471, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399,
    "compactness_se": 0.04904, "concavity_se": 0.05373, "concave_points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0, "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656, "concavity_worst": 0.7119, "concave_points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
}

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Machine Learning Diseases Prediction System',
                           ['Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity'],
                           default_index=0)

# Breast Cancer Prediction Page
if selected == "Breast Cancer Prediction":
    st.title('Breast Cancer Prediction using ML')

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        radius_mean = st.text_input('Radius Mean', value=str(sample_values['radius_mean']))
    with col2:
        texture_mean = st.text_input('Texture Mean', value=str(sample_values['texture_mean']))
    with col3:
        perimeter_mean = st.text_input('Perimeter Mean', value=str(sample_values['perimeter_mean']))
    with col4:
        area_mean = st.text_input('Area Mean', value=str(sample_values['area_mean']))
    with col5:
        smoothness_mean = st.text_input('Smoothness Mean', value=str(sample_values['smoothness_mean']))
       
    with col1:
        compactness_mean = st.text_input('Compactness Mean', value=str(sample_values['compactness_mean']))
    with col2:
        concavity_mean = st.text_input('Concavity Mean', value=str(sample_values['concavity_mean']))
    with col3:
        concave_points_mean = st.text_input('Concave Points Mean', value=str(sample_values['concave_points_mean']))
    with col4:
        symmetry_mean = st.text_input('Symmetry Mean', value=str(sample_values['symmetry_mean']))
    with col5:
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean', value=str(sample_values['fractal_dimension_mean']))
        
    with col1:
        radius_se = st.text_input('Radius SE', value=str(sample_values['radius_se']))
    with col2:
        texture_se = st.text_input('Texture SE', value=str(sample_values['texture_se']))
    with col3:
        perimeter_se = st.text_input('Perimeter SE', value=str(sample_values['perimeter_se']))
    with col4:
        area_se = st.text_input('Area SE', value=str(sample_values['area_se']))
    with col5:
        smoothness_se = st.text_input('Smoothness SE', value=str(sample_values['smoothness_se']))
         
    with col1:
        compactness_se = st.text_input('Compactness SE', value=str(sample_values['compactness_se']))
    with col2:
        concavity_se = st.text_input('Concavity SE', value=str(sample_values['concavity_se']))
    with col3:
        concave_points_se = st.text_input('Concave Points SE', value=str(sample_values['concave_points_se']))
    with col4:
        symmetry_se = st.text_input('Symmetry SE', value=str(sample_values['symmetry_se']))
    with col5:
        fractal_dimension_se = st.text_input('Fractal Dimension SE', value=str(sample_values['fractal_dimension_se']))
    
    with col1:
        radius_worst = st.text_input('Radius Worst', value=str(sample_values['radius_worst']))
    with col2:
        texture_worst = st.text_input('Texture Worst', value=str(sample_values['texture_worst']))
    with col3:
        perimeter_worst = st.text_input('Perimeter Worst', value=str(sample_values['perimeter_worst']))
    with col4:
        area_worst = st.text_input('Area Worst', value=str(sample_values['area_worst']))
    with col5:
        smoothness_worst = st.text_input('Smoothness Worst', value=str(sample_values['smoothness_worst']))
    
    with col1:
        compactness_worst = st.text_input('Compactness Worst', value=str(sample_values['compactness_worst']))
    with col2:
        concavity_worst = st.text_input('Concavity Worst', value=str(sample_values['concavity_worst']))
    with col3:
        concave_points_worst = st.text_input('Concave Points Worst', value=str(sample_values['concave_points_worst']))
    with col4:
        symmetry_worst = st.text_input('Symmetry Worst', value=str(sample_values['symmetry_worst']))
    with col5:
        fractal_dimension_worst = st.text_input('Fractal Dimension Worst', value=str(sample_values['fractal_dimension_worst']))

    # Select only the necessary features (23 features)
    user_input = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst
    ]

    breast_cancer_diagnosis = ''
    if st.button('Breast Cancer Test Result'):
        
        try:
            # Converting inputs to float
            user_input = [float(feature) for feature in user_input]

           
            
            breast_cancer_prediction = BreastCancer_model.predict([user_input])
            if breast_cancer_prediction[0] == 'M':
                breast_cancer_diagnosis = 'The person is likely to have benign breast cancer'
            else:
                breast_cancer_diagnosis = 'The person is likely to have malignant breast cancer'
            st.success(breast_cancer_diagnosis)
        except ValueError as ve:
            st.error(f'Please enter valid numbers for all fields. ValueError: {ve}')
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')

# Running the app
if __name__ == '__main__':
    st.success('Welcome to the Health Assistant application!')
