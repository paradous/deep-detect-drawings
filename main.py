
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from src import Image, Model

st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(page_title="Deep detect drawings")

model = Model()


# Header
st.title("Deep detect drawings")
st.write("""
Deep detect drawings is a Python app based on a CNN model made with PyTorch,
to recognize the item that you draw on the canvas. See the [notebook of the model's creation]
(https://colab.research.google.com/drive/1hh1lcDcXK3oxL2cEAPvlNXUY10bAhZp-?usp=sharing) for more details.
""")


# Drawing area
st.subheader("Drawing area")
st.markdown("Draw a banana or an axe. Don't make it easy !")
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#fff",
    background_color="#000",
    update_streamlit=True,
    drawing_mode="freedraw",
    key="canvas",
)
image = Image(canvas_result.image_data)


# Check if the user has written something
if (image.image is not None) and (not image.is_empty()):

    # Get the predicted class
    prediction = model.predict(image.get_prediction_ready())

    col3, col4 = st.beta_columns(2)

    # Display the image predicted by the model
    with col3:

        images = [
            "https://f003.backblazeb2.com/file/joffreybvn/deepdrawing/axe.png",
            "https://f003.backblazeb2.com/file/joffreybvn/deepdrawing/banana.png"
        ]

        st.subheader("Recognized image")
        st.markdown("The image recognized by the model")
        st.image(images[prediction], width=250)

    # Display the pro
    with col4:
        st.subheader("Probability distribution")
        st.markdown("Was your drawing hard to recognize ?")
        st.bar_chart(pd.DataFrame(
            model.probabilities,
            columns=["banana", "axe"]
        ).T)


# Sidebar
st.sidebar.header("About the author")
st.sidebar.markdown("""
**Joffrey Bienvenu**

Python dev, studying Machine Learning at BeCode.org.

 - Website: [joffreybvn.be](https://joffreybvn.be/)
 - Twitter: [@joffreybvn](https://twitter.com/Joffreybvn)
 - LinkedIn: [in/joffreybvn](https://www.linkedin.com/in/joffreybvn/)
 - Github: [joffreybvn](https://github.com/joffreybvn)
""")

st.sidebar.header("See on github")
st.sidebar.markdown("""
See the code and fork this project on Github:

[Deep Detect Drawings repository](https://github.com/Joffreybvn/deep-detect-drawings)
""")