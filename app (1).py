import detect as detect
import streamlit as st
import os

from PIL import Image

st.title('🌱 Weed Guard - Smart Weed Detection Using Computer Vision')
st.write('### This web app can detect whether a plant is a crop or weed.')
st.write('Made by Project Group 23')

st.write('#### Upload an image.')
uploaded_file = st.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

if uploaded_file is None:
    st.warning("No file has been uploaded.")
    st.stop()
else:
    if os.path.exists('runs/detect/exp'):
        os.rmdir('runs/detect/exp')

    image = Image.open(uploaded_file).convert("RGB")
    filename = str(uploaded_file.name)
    image = image.save(filename)

detect.run(weights='best.pt', conf_thres=0.1, source=filename)

st.image('runs/detect/exp/' + filename)

os.remove(filename)
os.remove('runs/detect/exp/' + filename)
