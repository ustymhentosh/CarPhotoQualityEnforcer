import streamlit as st
import numpy as np
import io
from PIL import Image
from car_models import (
    CarSide,
    CarOrNot,
    OverexposedOrNot,
    CarChopper,
    OveredarkenedOrNot,
)
from croper import Croper


st.set_page_config(layout="wide")

## ========== Models ==========


@st.cache_resource
def load_car_side_model():
    return CarSide("./models/side_detection_v0.pth")


@st.cache_resource
def load_caroper():
    return Croper()


@st.cache_resource
def load_car_or_not_model():
    return CarOrNot("./models/car_or_not_detection_v0.pth")


@st.cache_resource
def load_glare_model():
    return OverexposedOrNot()


@st.cache_resource
def load_position_model():
    return CarChopper()


@st.cache_resource
def load_dark_model():
    return OveredarkenedOrNot()


car_side_model = load_car_side_model()
car_or_not_model = load_car_or_not_model()
car_croper = load_caroper()
car_overexposed_model = load_glare_model()
car_position_model = load_position_model()
car_dark_model = load_dark_model()

## =====================

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
        image_data = bytes_image
        image_ = Image.open(io.BytesIO(image_data))
        image_ = image_.convert("RGB")
        no_car=False
        for i in range(1,5):
            if car_or_not_model.predict(image_) != "car" or car_position_model.crop(image_)[1] == "No car detected":
                # print("image rotated ", i, " times")
                # st.image(np.array(image_.rotate(i*90, expand=True)))
                if i == 1:
                    st.header("No car detected")
                    st.image(np.array(image_))
                if car_or_not_model.predict(image_.rotate(i*90, expand=True)) == "car" and car_position_model.crop(image_.rotate(i*90, expand=True))[1] != "No car detected":
                    st.header(f"Car detected after rotating the image")
                    print(f"Image rotated {i} times, car detected")
                    image = image_.rotate(i*90, expand=True)
                    st.image(np.array(image))
                    break
                elif i == 4:
                    print("image rotated ", i, " times, no car detected")
                    # st.header("No car detected")
                    no_car = True
                    # st.image(np.array(image_))
                    image = image_
            else:
                image = image_
                st.image(np.array(image))
                break

with col2:
    st.header("Assesment")
    st.write("")

    proceed = False
    if uploaded_file:
        # if image:
        #     image_data = bytes_image
        #     image = Image.open(io.BytesIO(image_data))
        #     image = image.convert("RGB")
        proceed = True

    ## Car not car
    st.write(f"#### Basic Car Check")
    st.write("This test checks if there is a **car** in your photo")
    if proceed:
        car_or_not_prediction = car_or_not_model.predict(image)
        if car_or_not_prediction == "car":
            if no_car:
                st.warning("Car found, but not detected", icon="️⚠️")
            else:
                st.success("Success - car found", icon="✅")
        else:
            st.error("Fail - no car found", icon="🚨")
        print(f"Ustyms: {car_or_not_prediction}, \n YOLO: {car_position_model.crop(image)[1]}")

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
        if car_side_prediction in [
            "angle-front-on-left",
            "front",
            "angle-front-on-right",
        ]:
            st.success(f"Success - right angle {car_side_prediction}", icon="✅")
        else:
            st.warning(
                f"Not recommended angle for cover photo - {car_side_prediction}",
                icon="⚠️",
            )

    ## Glare
    # st.write(f"#### Glare Check")
    # st.write("This test looks for **glare** in your photo")
    # if proceed:
    #     car_overexposed_or_not = car_overexposed_model.predict(
    #         Image.fromarray(croped_image_arr)
    #     )
    #     print(car_overexposed_or_not)
    #     if car_overexposed_or_not == "not_overexposed":
    #         st.success("Success - photo is NOT overexposed", icon="✅")
    #         proceed = True
    #     else:
    #         st.error("Fail - photo is overexposed", icon="🚨")
    #         proceed = True

    ## Exposure Check
    st.write(f"#### Exposure Check")
    st.write(
        "This test checks for **underexposure** and **overexposure** in your photo"
    )

    if proceed:
        # Predict both overexposure and underexposure
        is_overexposed = car_overexposed_model.predict(
            Image.fromarray(croped_image_arr)
        )
        is_too_dark = car_dark_model.predict(Image.fromarray(croped_image_arr))

        print(f"Overexposed model: {is_overexposed}, Dark model: {is_too_dark}")

        # Logic for determining exposure status
        if is_overexposed == "not_overexposed" and is_too_dark == "not_too_dark":
            st.success("Success - photo has good exposure", icon="✅")
            proceed = True
        elif is_overexposed != "not_overexposed":
            st.warning("Warning - photo is overexposed", icon="⚠️")
            proceed = True
        elif is_too_dark != "not_too_dark":
            st.warning("Warning - photo is too dark", icon="⚠️")
            proceed = True

    ## Positioning
    st.write(f"#### Car Position Check")
    st.write("This test looks for **position** of your car on a photo")
    if proceed:
        new_image, car_position_result = car_position_model.crop(image)
        print(car_position_result)
        if car_position_result == "Success":
            if not no_car:
                st.success(car_position_result, icon="✅")
            else:
                st.warning("No car detected", icon="⚠️")
            with col1:
                if not no_car:
                    st.write(f"#### Final Image")
                    st.image(np.array(new_image))
        else:
            if not no_car:
                st.warning(car_position_result, icon="⚠️")
            else:
                st.warning("No car detected", icon="⚠️")
            with col1:
                if not no_car:
                    st.write(f"#### Recommended Image Expansion")
                    st.image(np.array(new_image))


    ### Quality Test
    st.write(f"#### Quality Check")
    st.write("This test gives you information about the **quality** of your image")
    if proceed:
        print(new_image.size[0], new_image.size[1])
        if new_image.size[0] < 1800 and new_image.size[1] < 1200:
            st.warning(
                f"Your image has size {new_image.size}, we recommend to have at least (1800, 1200) ",
                icon="⚠️",
            )
        else:
            st.success("Good Quality", icon="✅")
