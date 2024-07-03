import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

model = load_model('corals.h5')


st.set_page_config(
    page_title="Identify Healthy Corals",
    page_icon = ":coral:",
    initial_sidebar_state = 'auto'
)

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

with st.sidebar:
        st.image('corals.jpg')
        st.title("Corals")
        st.subheader("Predicts whether a coral is healthy or bleached")

st.write("""
         # Healthy Coral Identification
         """)

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (160,160)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)

    class_names = ['bleached_corals', 'healthy_corals']

    if predictions < 0.5:
      st.markdown("A bleached coral !")
    else :
      st.markdown("A healthy coral :)")

st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache_data(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('corals.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()