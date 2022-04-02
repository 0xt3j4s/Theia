from cProfile import label
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

def get_label(img):
    model = keras.models.load_model('keras_model.h5', compile= False)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = img

    # image = Image.open(img)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    return np.argmax(prediction)




st.title("Theia | DimensionEd")
st.header("Image Recognition")
st.text("Upload an image and find the ar model")
# file upload and handling logic
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
#image = Image.open(img_name).convert('RGB')
    st.image(image, caption='Uploaded a X Ray IMage.', use_column_width=True)
    st.write("")
    st.write("Classifying the image .........hold tight")
    label = get_label(image)
    if label == 0:
        st.write("-------> stomach")
    elif label == 1:
        st.write("-------> black hole")
    elif label == 2:
        st.write("-------> cyclone")
    elif label == 3:
        st.write("-------> earth")
    elif label == 4:
        st.write("-------> lungs")
    elif label == 5:
        st.write("-------> volcano")


        

    labels = open('labels.txt', 'r')
    l = labels.read()

    temp = list(l.split('\n'))
    print(f"label = {label}")
    print(f"temp = {temp}")
