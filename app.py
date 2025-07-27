import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cnn_model.h5")

model = load_model()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Judul Aplikasi
st.title("Klasifikasi Gambar CIFAR-10 dengan CNN")
st.write("Upload gambar untuk melihat hasil prediksi klasifikasi.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((32, 32))
    st.image(img, caption='Gambar yang diupload', use_column_width=True)

    # Preprocessing
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]

    # Output
    st.subheader("Hasil Prediksi:")
    st.write(f"**Kelas:** {class_names[class_id]}")
    st.write(f"**Confidence:** {confidence:.2f}")
