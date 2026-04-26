import zipfile
from io import BytesIO

import cv2
import numpy as np
import streamlit as st


st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        .header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .colab-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.7rem 1rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #1a73e8, #0b57d0);
            color: white !important;
            font-weight: 700;
            text-decoration: none !important;
            box-shadow: 0 8px 20px rgba(26, 115, 232, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.15);
            white-space: nowrap;
        }
        .colab-button:hover {
            filter: brightness(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("RGB2GRAY AND GRAY2RGB")
with header_col2:
    st.markdown(
        '<div style="height:0.4rem"></div>'
        '<a class="colab-button" href="https://colab.research.google.com/drive/1kgTiQOitfd91CEfvY4ILY3SFzO9OvzSV?usp=sharing" target="_blank" rel="noopener noreferrer">Open in Colab</a>',
        unsafe_allow_html=True,
    )


def process_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    r_channel = gray
    g_channel = (gray * 0.7).astype(np.uint8)
    b_channel = (gray * 0.4).astype(np.uint8)
    fake_color = cv2.merge([r_channel, g_channel, b_channel])

    color_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    colorized = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    return img, gray, fake_color, color_map, colorized


def render_image_row(file_name: str, img: np.ndarray) -> None:
    original, gray, fake_color, color_map, colorized = process_image(img)

    st.markdown("---")
    st.subheader(f"Image: {file_name}")

    header_cols = st.columns(5)
    for column, label in zip(
        header_cols,
        ["Original", "Gray", "Fake Color", "Color Map", "Fast Colorized"],
    ):
        with column:
            st.markdown(f"**{label}**")

    image_cols = st.columns(5)

    with image_cols[0]:
        st.image(original, width="stretch")

    with image_cols[1]:
        st.image(gray, clamp=True, width="stretch")

    with image_cols[2]:
        st.image(fake_color, width="stretch")

    with image_cols[3]:
        st.image(color_map, width="stretch")

    with image_cols[4]:
        st.image(colorized, width="stretch")


uploaded_file = st.file_uploader(
    "Upload a ZIP or a single image",
    type=["zip", "png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.read()

    if file_name.lower().endswith(".zip"):
        zip_file = zipfile.ZipFile(BytesIO(file_bytes))
        image_files = [
            image_name
            for image_name in zip_file.namelist()
            if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

        st.success(f"Total Images Found: {len(image_files)}")

        if not image_files:
            st.info("No supported image files were found in the ZIP.")

        for image_name in image_files:
            np_arr = np.frombuffer(zip_file.read(image_name), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                st.warning(f"Skipping {image_name}")
                continue

            render_image_row(image_name, img)
    else:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            st.error("The uploaded file could not be read as an image.")
        else:
            st.success("Single image loaded successfully.")
            render_image_row(file_name, img)