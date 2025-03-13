import os
import glob
import re
import random
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")

st.title("Pneumonia Detection")
st.markdown("#### Use the buttons below to select or upload a image.")

# Load & incorporate model
@st.cache_resource
def load_pneumonia_model():

    def extract_file_number(filename):
        match = re.search(r'model_part_(\d+)\.pth', filename)
        if match:
            return int(match.group(1))
        return 0

    model_parts_path = "./data/neural_network/model_part_*"
    output_model_path = "./data/neural_network/pneumonia_model.h5"

    try:
        # Model existed on cache
        if os.path.exists(output_model_path):
            print("Loading existing model...")
            model = load_model(output_model_path)
            return model

        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        parts = sorted(glob.glob(model_parts_path), key=extract_file_number)
        if not parts:
            print(f"No model parts found at {model_parts_path}")
            return None
            
        st.info(f"Merging {len(parts)} model parts...")
        with open(output_model_path, "wb") as output_file:
            for part in parts:
                with open(part, "rb") as chunk:
                    output_file.write(chunk.read())
                    
        model = load_model(output_model_path)
        print("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_image(img):
    # Convert grayscale to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize((224, 224))
    
    # Normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

# Predictions
def predict_pneumonia(model, img_array):
    prediction = model.predict(img_array)
    if isinstance(prediction, list):
        prediction = prediction[0]
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        return prediction[0]
    else:
        return prediction[0][0]

def get_random_image(folder_path):
    if os.path.exists(folder_path):
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            selected_image = random.choice(images)
            img_path = os.path.join(folder_path, selected_image)
            return Image.open(img_path), selected_image
    return None, None

def main():
    if 'current_img' not in st.session_state:
        st.session_state.current_img = None
    if 'image_source' not in st.session_state:
        st.session_state.image_source = None
    if 'image_name' not in st.session_state:
        st.session_state.image_name = None
    
    model = load_pneumonia_model()
    
    if model is None:
        st.warning("Model not found !!!")
        return

    folder_path_normal = './data/neural_network/chest_xray/test/NORMAL'
    folder_path_pneumonia = './data/neural_network/chest_xray/test/PNEUMONIA'
    
    if not (os.path.exists(folder_path_normal) and os.path.exists(folder_path_pneumonia)):
        st.error(f"Folder not found: '{folder_path_normal}' or '{folder_path_pneumonia}'")
        return
    
    button_row = st.columns(3)
    
    # Button to select random image
    with button_row[0]:
        if st.button("Random", use_container_width=1):
            folder = random.choice(['NORMAL', 'PNEUMONIA'])
            selected_folder = folder_path_normal if folder == 'NORMAL' else folder_path_pneumonia
            st.session_state.current_img, st.session_state.image_name = get_random_image(selected_folder)
            st.session_state.image_source = folder
    
    # Button to select normal image
    with button_row[1]:
        if st.button("Normal", use_container_width=1):
            st.session_state.current_img, st.session_state.image_name = get_random_image(folder_path_normal)
            st.session_state.image_source = "NORMAL"
    
    # Button to select pneumonia image
    with button_row[2]:
        if st.button("Pneumonia", use_container_width=1):
            st.session_state.current_img, st.session_state.image_name = get_random_image(folder_path_pneumonia)
            st.session_state.image_source = "PNEUMONIA"
    
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    
    # Handle uploaded file
    if uploaded_file is not None:
        try:
            st.session_state.current_img = Image.open(uploaded_file)
            st.session_state.image_source = "UPLOADED"
            st.session_state.image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error opening image: {e}")
    
    # Display the current image
    if st.session_state.current_img is not None:
        caption = f"Selected X-ray: {st.session_state.image_name}"
        if st.session_state.image_source and st.session_state.image_source != "UPLOADED":
            caption += f" (Category: {st.session_state.image_source})"
        st.image(st.session_state.current_img, caption=caption, use_container_width=True)
    
    if st.session_state.current_img is not None:
        if st.button("Analyze Image", use_container_width=True):
            with st.spinner("Analyzing..."):
                processed_img = preprocess_image(st.session_state.current_img)
                
                try:
                    prediction_result = predict_pneumonia(model, processed_img)
                    
                    if isinstance(prediction_result, np.ndarray) and len(prediction_result) > 1:
                        class_names = ["Normal", "Pneumonia"]
                        predicted_class = np.argmax(prediction_result) #getmax
                        confidence = prediction_result[predicted_class] * 100
                        
                        if predicted_class == 1:  # Pneumonia ([1])
                            st.error(f"Pneumonia Detected (Confidence: {confidence:.2f}%)")
                        else:  # Normal ([0])
                            st.success(f"No Pneumonia Detected (Confidence: {confidence:.2f}%)")
                            
                        st.write("### Detailed Results:")
                        for i, class_name in enumerate(class_names):
                            st.write(f"{class_name}: {prediction_result[i] * 100:.2f}%")
                    else:
                        # not in nparg arr property do it as binary crossentropy
                        if prediction_result > 0.5: 
                            pneumonia_probability = prediction_result * 100
                            st.error(f"Pneumonia Detected (Confidence: {pneumonia_probability:.2f}%)")
                        else:
                            normal_probability = (1 - prediction_result) * 100
                            st.success(f"No Pneumonia Detected (Confidence: {normal_probability:.2f}%)")
                        
                        st.write("### Detailed Results:")
                        st.write(f"Normal: {(1 - prediction_result) * 100:.2f}%")
                        st.write(f"Pneumonia: {prediction_result * 100:.2f}%")
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("Try to check the model architecture and ensure the preprocessing matches the expected input.")
                
                # Compare with actual in local state
                if st.session_state.image_source in ["NORMAL", "PNEUMONIA"]:
                    expected_label = "Normal" if st.session_state.image_source == "NORMAL" else "Pneumonia"
                    predicted_label = "Normal" if (isinstance(prediction_result, np.ndarray) and np.argmax(prediction_result) == 0) or (not isinstance(prediction_result, np.ndarray) and prediction_result <= 0.5) else "Pneumonia"
                    
                    if expected_label == predicted_label:
                        st.success(f"Model correctly identified the image as {predicted_label} :)")
                    else:
                        st.error(f"Model incorrectly classified the image :(")
                        st.markdown(f"""
                            ```c
                            Expected  : {expected_label + ' ' * (15 - len(expected_label))}
                            Predicted : {predicted_label + ' ' * (15 - len(predicted_label))}
                            ```
                        """)
    else:
        st.info("Please select or upload an X-ray image to analyze.")
    
    # Information section
    st.sidebar.title("About")
    st.sidebar.info("""
    **How to use:**
    1. Select an X-ray image:
       - "Random" for a random sample
       - "Normal" for a sample without pneumonia
       - "Pneumonia" for a sample with pneumonia
       - **Or upload your own image**
    2. Click "Analyze Image" to check for pneumonia
    3. View the results and probability scores
    
    **Important:** This tool is for demonstration purposes only!!!!
    """)

if __name__ == "__main__":
    main()