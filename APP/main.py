import streamlit as st
import numpy as np
import io
from PIL import Image
from car_models import CarSide, CarOrNot
from croper import Croper


st.set_page_config(layout="wide")

## == Models ==


@st.cache_resource
def load_car_side_model():
    return CarSide("./side_detection_v0.pth")


@st.cache_resource
def load_caroper():
    return Croper()


@st.cache_resource
def load_car_or_not_model():
    return CarOrNot("./car_or_not_detection_v0.pth")


car_side_model = load_car_side_model()
car_or_not_model = load_car_or_not_model()
car_croper = load_caroper()

st.write(
    """
    # :blue[Make a good quality photo]
"""
)

col1, col2 = st.columns(2)

with col1:
    st.header("Your photo")
    uploaded_file = st.file_uploader(
        "You can re-upload to pass all tests", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        bytes_image = uploaded_file.getvalue()
        st.image(bytes_image)

with col2:
    st.header("Assesment")
    st.write("")

    proceed = False
    if uploaded_file:
        image_data = bytes_image
        image = Image.open(io.BytesIO(image_data))
        proceed = True

    ## Car not car
    st.write(f"#### Basic Car Check")
    st.write("This test checks if there is a **car** in your photo")
    if proceed:
        car_or_not_prediction = car_or_not_model.predict(image)
        if car_or_not_prediction == "car":
            st.success("Success - car found", icon="✅")
            # st.badge("Success - car found", icon="✔", color="green")
            proceed = True
        else:
            st.error("Fail - no car found", icon="🚨")
            # st.badge("Fail - no car found", icon="✖", color="red")
            proceed = False


    ### Quality Test
    st.write(f"#### Quality Check")
    st.write("This test gives you information about the **quality** of your image")
    if proceed:
        st.info(f"Your image has size {image.size}", icon="ℹ️")

    ## Croping Proces
    if proceed:
        img_arr = np.array(image)
        croped_image_arr = car_croper.get_croped(img_arr)

    ## Side prediction
    st.write(f"#### Photo Angle Check")
    st.write("This test tells if the photo is taken from the best possible **angle**")
    if proceed:
        car_side_prediction = car_side_model.predict(Image.fromarray(croped_image_arr))
        print(car_side_prediction)
        if car_side_prediction in ["angle-front-on-left", "front"]:
            st.success("Success - right angle", icon="✅")
            # st.badge("Success - right angle", icon="✔", color="green")
            proceed = True
        else:
            st.error("Fail - bad angle", icon="🚨")
            # st.badge("Fail - bad angle", icon="✖", color="red")
            proceed = False

    ## Positioning
    ## Glare
