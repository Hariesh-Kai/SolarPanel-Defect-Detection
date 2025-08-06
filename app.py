import streamlit as st
import requests
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf

# ==== Streamlit Config ====
st.set_page_config(page_title="SolarGuard", page_icon="⚡")
st.title("🔍 SolarGuard - Solar Panel AI")
st.sidebar.title("Mode")
mode = st.sidebar.radio("Choose Phase", ["Classification", "Detection"])

# ==== Shared Upload ====
uploaded = st.file_uploader("Upload a solar panel image", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Input Image", use_column_width=True)

# ==== Phase 1: Classification ====
if mode == "Classification":
    st.subheader("Classification Mode")
    # Load once
    @st.cache_resource
    def load_classifier():
        model = tf.keras.models.load_model("models/MobileNetV2_solar_model.h5", compile=False)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    clf = load_classifier()

    # Preprocess
    img = image.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, 0)

    # Predict
    preds = clf.predict(arr)[0]
    classes = ['Clean','Dusty','Bird-Drop','Electrical-Damage','Physical-Damage','Snow-Covered']
    idx = int(np.argmax(preds))
    label = classes[idx]
    conf  = preds[idx]

    st.markdown(f"### Prediction: **{label}**  ({conf*100:.2f}% confidence)")

# ==== Phase 2: Detection ====
else:
    st.subheader("Detection Mode")
    # Roboflow config
    API_KEY   = "SuUAkpr94QtUUsvm0BHc"
    MODEL_ID  = "faulty_solar_panel-giwvx"
    VERSION   = 2
    URL       = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={API_KEY}"
    COLORS    = {
        'Clean':'green','Dusty':'orange','Bird-Drop':'blue',
        'Electrical-Damage':'purple','Physical-Damage':'red','Snow-Covered':'cyan'
    }

    # Call API
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    resp = requests.post(URL, files={"file":buf.getvalue()})
    if resp.status_code != 200:
        st.error(f"❌ Roboflow error {resp.status_code}")
        st.stop()
    data = resp.json()
    preds = data.get("predictions", [])
    if not preds:
        st.warning("No defects detected.")
        st.stop()

    # Draw boxes
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for p in preds:
        x,y,w,h = p["x"],p["y"],p["width"],p["height"]
        cls,conf = p["class"],p["confidence"]
        x0,y0 = int(x-w/2), int(y-h/2)
        x1,y1 = int(x+w/2), int(y+h/2)
        c = COLORS.get(cls,"red")
        draw.rectangle([x0,y0,x1,y1], outline=c, width=3)

        label = f"{cls} ({conf:.2f})"
        bbox = draw.textbbox((0,0),label,font=font)
        tw,th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        lx0,ly0 = x0+2, y0+2
        draw.rectangle([lx0,ly0,lx0+tw+4, ly0+th+4], fill=c)
        draw.text((lx0+2, ly0+2), label, fill="white", font=font)

    st.success(f"✅ Detected {len(preds)} objects")
    st.image(image, caption="Detections", use_column_width=True)
