
import streamlit as st
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="SmartMove",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 SmartMove")
st.subheader("Human Activity Recognition Using LSTM")

st.write(
    "Classify smartphone motion sensor sequences "
    "using a trained LSTM deep learning model."
)

st.info(
    "Upload your trained model and compatible sensor "
    "data to enable predictions."
)

model_file = st.file_uploader(
    "Upload trained Keras model",
    type=["keras"]
)

data_file = st.file_uploader(
    "Upload test sensor data (.npy)",
    type=["npy"]
)

activities = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

if model_file and data_file:
    model = tf.keras.models.load_model(model_file)
    samples = np.load(data_file)

    if samples.ndim == 2:
        samples = samples[np.newaxis, ...]

    if samples.ndim != 3 or samples.shape[1:] != (128, 9):
        st.error("Expected data shape: (samples, 128, 9).")
    else:
        index = st.number_input(
            "Sample index",
            min_value=0,
            max_value=len(samples) - 1,
            value=0
        )

        if st.button("Predict Activity"):
            scores = model.predict(
                samples[int(index):int(index) + 1],
                verbose=0
            )[0]

            best = int(np.argmax(scores))

            st.success(
                f"Predicted activity: {activities[best]}"
            )

            st.metric(
                "Model confidence",
                f"{scores[best] * 100:.2f}%"
            )

            st.bar_chart({
                "Prediction score": dict(zip(activities, scores))
            })
