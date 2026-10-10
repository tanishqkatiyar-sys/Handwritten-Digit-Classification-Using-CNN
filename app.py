import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Digit AI Studio",
    page_icon="🔢",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #172554);
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #93c5fd;
}
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🔢 Digit AI Studio</div>',
    unsafe_allow_html=True
)
st.caption("Upload → crop → preprocess → predict")


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "Perceptron": load_model("perceptron.keras"),
        "ANN": load_model("ann.keras"),
        "CNN": load_model("cnn.keras")
    }


try:
    models = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.info(
        "Keep perceptron.keras, ann.keras and cnn.keras "
        "in the same folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess_image(
    uploaded_image,
    crop_left,
    crop_top,
    crop_right,
    crop_bottom,
    threshold,
    padding_percent
):
    img = ImageOps.exif_transpose(
        uploaded_image
    ).convert("RGB")

    width, height = img.size

    # Crop coordinates are percentages of the full image
    left = int(width * crop_left / 100)
    top = int(height * crop_top / 100)
    right = int(width * crop_right / 100)
    bottom = int(height * crop_bottom / 100)

    if right <= left or bottom <= top:
        raise ValueError("Please select a valid crop area.")

    crop = img.crop((left, top, right, bottom))

    rgb = np.asarray(crop, dtype=np.float32)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # --------------------------------------------------
    # DETECT BLUE PEN INK
    # --------------------------------------------------
    blue_score = blue - red

    blue_mask = (
        (blue_score > 15)
        & (blue > green * 0.90)
        & (blue < 245)
        & (red < 210)
    )

    blue_pixel_count = np.count_nonzero(blue_mask)

    if blue_pixel_count > 20:
        # White digit ink on a black background
        foreground = blue_mask.astype(np.uint8) * 255

    else:
        # Fallback for black or pencil handwriting
        gray_img = ImageOps.grayscale(crop)
        gray_img = ImageOps.autocontrast(gray_img)

        gray = np.asarray(gray_img, dtype=np.float32)

        blurred_img = gray_img.filter(
            ImageFilter.GaussianBlur(radius=15)
        )
        background = np.asarray(
            blurred_img, dtype=np.float32
        )

        contrast = background - gray

        foreground = (
            contrast > threshold
        ).astype(np.uint8) * 255

    # Remove isolated noise, but do not select only
    # the largest connected component.
    foreground_img = Image.fromarray(foreground)
    foreground_img = foreground_img.filter(
        ImageFilter.MedianFilter(size=3)
    )
    foreground = np.asarray(foreground_img)

    # Find all foreground pixels together
    ys, xs = np.where(foreground > 0)

    if len(xs) < 10:
        raise ValueError(
            "Digit not detected. Expand the crop or lower the threshold."
        )

    # Bounding box of ALL detected digit pixels
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    digit = foreground[y1:y2, x1:x2]

    # --------------------------------------------------
    # ADD PADDING AROUND THE COMPLETE DIGIT
    # --------------------------------------------------
    digit_h, digit_w = digit.shape

    pad_y = max(2, int(digit_h * padding_percent / 100))
    pad_x = max(2, int(digit_w * padding_percent / 100))

    padded = np.pad(
        digit,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=0
    )

    # --------------------------------------------------
    # RESIZE PROPORTIONALLY TO FIT 20 x 20
    # --------------------------------------------------
    digit_img = Image.fromarray(padded)
    digit_img.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Center on the black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)

    canvas.paste(
        digit_img,
        (
            (28 - digit_img.width) // 2,
            (28 - digit_img.height) // 2
        )
    )

    final_image = (
        np.asarray(canvas, dtype=np.float32) / 255.0
    )

    return crop, foreground, final_image


# --------------------------------------------------
# PREDICT
# --------------------------------------------------
def predict_digit(model, image):
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) == 4:
        model_input = image.reshape(1, 28, 28, 1)
    elif len(shape) == 3:
        model_input = image.reshape(1, 28, 28)
    elif len(shape) == 2:
        model_input = image.reshape(1, 784)
    else:
        raise ValueError(
            f"Unsupported model input shape: {shape}"
        )

    scores = np.asarray(
        model.predict(model_input, verbose=0)[0]
    )

    if scores.size != 10:
        raise ValueError(
            "Expected 10 output scores for digits 0–9."
        )

    digit = int(np.argmax(scores))
    score = float(scores[digit]) * 100

    return digit, score, scores


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:

    original = ImageOps.exif_transpose(
        Image.open(uploaded_file)
    ).convert("RGB")

    st.subheader("1. Original image")
    st.image(original, use_container_width=True)

    st.subheader("2. Select the complete digit")

    st.info(
        "Move the sliders so the crop includes the entire digit, "
        "including the top loop and bottom stroke. Avoid cutting "
        "off any part of the digit."
    )

    c1, c2 = st.columns(2)

    with c1:
        crop_left = st.slider(
            "Crop left (%)",
            0, 95, 15,
            key="crop_left"
        )

        crop_top = st.slider(
            "Crop top (%)",
            0, 95, 15,
            key="crop_top"
        )

    with c2:
        crop_right = st.slider(
            "Crop right (%)",
            5, 100, 90,
            key="crop_right"
        )

        crop_bottom = st.slider(
            "Crop bottom (%)",
            5, 100, 90,
            key="crop_bottom"
        )

    threshold = st.slider(
        "Threshold for non-blue handwriting",
        5, 80, 25
    )

    padding_percent = st.slider(
        "Padding around digit (%)",
        5, 40, 15
    )

    try:
        cropped, foreground, processed = preprocess_image(
            original,
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
            threshold,
            padding_percent
        )

        st.subheader("3. Preprocessing result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                cropped,
                caption="Selected crop",
                use_container_width=True
            )

        with col2:
            st.image(
                foreground,
                caption="Detected ink",
                clamp=True,
                use_container_width=True
            )

        with col3:
            st.image(
                processed,
                caption="Final 28 × 28 model input",
                clamp=True,
                use_container_width=True
            )

        st.warning(
            "Check that the final image shows the complete digit. "
            "If the top loop or bottom stroke is missing, adjust "
            "the crop sliders before checking predictions."
        )

        # --------------------------------------------------
        # MODEL RESULTS
        # --------------------------------------------------
        st.subheader("4. Model predictions")

        results = {}

        for name, model in models.items():
            digit, score, scores = predict_digit(
                model, processed
            )

            results[name] = {
                "digit": digit,
                "score": score,
                "scores": scores
            }

        columns = st.columns(3)

        for col, (name, result) in zip(
            columns, results.items()
        ):
            with col:
                st.metric(
                    name,
                    f"Digit {result['digit']}",
                    f"{result['score']:.2f}% output score"
                )

        predicted_digits = [
            result["digit"]
            for result in results.values()
        ]

        if len(set(predicted_digits)) == 1:
            st.success(
                f"All three models predict {predicted_digits[0]}."
            )
        else:
            st.warning(
                "The models disagree. Check the processed image "
                "and your models' training preprocessing."
            )

        # --------------------------------------------------
        # SCORE TABLE
        # --------------------------------------------------
        st.subheader("5. Scores for digits 0–9")

        score_df = pd.DataFrame({
            name: result["scores"] * 100
            for name, result in results.items()
        })

        score_df.index.name = "Digit"

        st.dataframe(
            score_df.round(2),
            use_container_width=True
        )

        st.caption(
            "Output scores are not guaranteed to be calibrated "
            "probabilities or correct predictions."
        )

    except Exception as e:
        st.error(f"Preprocessing or prediction failed: {e}")

else:
    st.info("Upload an image to begin.")

st.markdown("---")
st.caption("Built with Streamlit, TensorFlow, Pillow and NumPy.")
