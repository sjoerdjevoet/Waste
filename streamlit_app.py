import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import json
import time
import random
import math

# ---- Config paths (pas aan indien nodig) ----
IMAGE_PATH = "./image_1715161701.865404.png"
ANNOTATION_PATH = "./image_1715161701.865404.json"

# ---- Vaste metadata (blijft altijd gelijk) ----
metadata = {
    "source_experiment": "exp_20230715_1423",
    "validation_loss": 0.2543,
    "promotion_time": "2025-07-15T14:23:00",
    "metrics": {"training_time": 15670},  # in seconden (=> 94.5 min)
    "model_type": "Faster R-CNN",
    "classes": ["PCB", "CAPACITOR", "ALUMINIUM"],
}

# ---- Streamlit pagina config ----
st.set_page_config(
    page_title="Waste Detection AI (Demo)",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Helpers ----
def format_timestamp(timestamp: str) -> str:
    # indien ISO string, toon leesbare vorm; anders toon zoals gegeven
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y, %H:%M:%S")
    except Exception:
        return timestamp

@st.cache_data
def load_annotation(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_candidate_scores(threshold: float):
    """
    Bouw de discrete kandidaatwaarden: threshold, threshold+0.1, ..., 0.9
    Returnt een lijst met floats (bijv. [0.7,0.8,0.9]) of [] als threshold>0.9
    """
    # stap van 0.1; rond af naar 1 decimaal
    first = math.ceil(threshold * 10) / 10.0
    candidates = []
    # integer loop van first*10 tot 9
    start = int(first * 10)
    for i in range(start, 10):  # 0..9 -> 0.0..0.9
        val = round(i / 10.0, 1)
        if val <= 0.9:
            candidates.append(val)
    return candidates

def discrete_fake_score_for_all(n_boxes: int, threshold: float):
    """
    Geef n_boxes scores: elk random gekozen uit kandidaten >= threshold.
    Als er geen kandidaten (threshold > 0.9), geef dan threshold clipped to <=0.99.
    """
    candidates = make_candidate_scores(threshold)
    if candidates:
        # kies onafhankelijk per box (met replacement)
        return [random.choice(candidates) for _ in range(n_boxes)]
    else:
        # threshold > 0.9: gebruik clipped value per box
        val = round(min(0.99, threshold), 2)
        return [val for _ in range(n_boxes)]

# ---- Detectie: laad JSON en creëer fake scores >= threshold ----
def detect_objects(confidence_threshold=0.7):
    data = load_annotation(ANNOTATION_PATH)
    shapes = data.get("shapes", [])

    boxes = []
    labels = []
    # label -> integer mapping (zelfde mapping in draw & table)
    label_map = {"PCB": 1, "CAPACITOR": 2, "ALUMINIUM": 3}

    for shape in shapes:
        (x1, y1), (x2, y2) = shape["points"]
        boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
        labels.append(torch.tensor(label_map.get(shape["label"], 0)))

    # genereer discrete random scores ≥ threshold
    scores = discrete_fake_score_for_all(len(boxes), confidence_threshold)

    return boxes, scores, labels

# ---- Tekenen ----
def draw_boxes(image: Image.Image, boxes, scores, labels):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    rev_map = {1: "PCB", 2: "CAPACITOR", 3: "ALUMINIUM"}

    for box, score, label in zip(boxes, scores, labels):
        # convert tensor -> numpy array
        box_arr = box.cpu().numpy() if hasattr(box, "cpu") else box
        x1, y1, x2, y2 = box_arr
        draw.rectangle([(x1, y1), (x2, y2)], outline="#FF4B4B", width=3)

        text = f"{rev_map.get(label.item() if hasattr(label, 'item') else label, 'UNK')} {score:.2f}"
        try:
            tb = draw.textbbox((x1, y1), text, font=font)
            txt_w = tb[2] - tb[0]
            txt_h = tb[3] - tb[1]
        except AttributeError:
            txt_w, txt_h = draw.textsize(text, font=font)

        padding = 4
        draw.rectangle([(x1 - padding, y1 - padding), (x1 + txt_w + padding, y1 + txt_h + padding)], fill="#FF4B4B")
        draw.text((x1, y1), text, fill="white", font=font)

    return image

# ---- UI ----
def main():
    # styling (kort)
    st.markdown(
        """
        <style>
            .stApp { background-color: #f8f9fa; }
            .stButton>button { background-color: #FF4B4B; color: white; font-weight: bold; }
            .model-card { background-color: white; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
            .metric-box { background-color: #f0f2f6; padding: 0.6rem; border-radius: 6px; }
            .header-text { color: #FF4B4B; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    

    # ✅ title at the very top
st.markdown(
    "<h1 style='text-align:center; color:#FF4B4B;'>♻️ Waste Detection AI (Demo)</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#666;'>Upload een afbeelding met afval en het model geeft een voorspelling</p>",
    unsafe_allow_html=True
)

# metadata card (fixed, styled)
with st.container():
    st.markdown('<div class="model-card">', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("""
        <h3 style="color:#FF4B4B; margin-bottom:0.5rem;">Production Model</h3>
        <div style="font-weight:bold;">Faster R-CNN</div>
        <div style="font-size:0.85rem; color:#666;">Object Detection</div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:0.5rem;">
            <div style="background:#f9f9f9; padding:6px 10px; border-radius:8px; border:1px solid #eee;">
                <div style="font-size:0.8rem; color:#666;">Experiment ID</div>
                <div style="font-weight:bold;">{metadata['source_experiment']}</div>
            </div>
            <div style="background:#f9f9f9; padding:6px 10px; border-radius:8px; border:1px solid #eee;">
                <div style="font-size:0.8rem; color:#666;">Validation Loss</div>
                <div style="font-weight:bold;">{metadata['validation_loss']:.4f}</div>
            </div>
            <div style="background:#f9f9f9; padding:6px 10px; border-radius:8px; border:1px solid #eee;">
                <div style="font-size:0.8rem; color:#666;">Last Updated</div>
                <div style="font-weight:bold;">{format_timestamp(metadata['promotion_time'])}</div>
            </div>
            <div style="background:#f9f9f9; padding:6px 10px; border-radius:8px; border:1px solid #eee;">
                <div style="font-size:0.8rem; color:#666;">Training Time</div>
                <div style="font-weight:bold;">{metadata['metrics']['training_time']/60:.1f} min</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# main layout
left, right = st.columns(2)

with left:
    uploaded = st.file_uploader("Kies een afbeelding (optioneel)", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.7, step=0.1,
                          help="Geef aan vanaf welke confidence level je de voorspellingen wilt zien")
    if uploaded is not None:
        st.image(uploaded, caption="Geüploade afbeelding (wordt niet gebruikt voor detectie)", use_container_width=True)

with right:
    if uploaded and st.button("Detect Waste", type="primary"):
        with st.spinner("Analyzing (demo)..."):
            start = time.time()
            demo_img = Image.open(IMAGE_PATH).convert("RGB")
            boxes, scores, labels = detect_objects(confidence_threshold=threshold)

            filtered = [(b, s, l) for b, s, l in zip(boxes, scores, labels) if s >= threshold]
            if filtered:
                boxes, scores, labels = zip(*filtered)
            else:
                boxes, scores, labels = [], [], []

            result = draw_boxes(demo_img.copy(), boxes, scores, labels)
            st.image(result, caption="Detection Results (demo)", use_container_width=True)

            elapsed = time.time() - start
            st.success(f"Detection finished in {elapsed:.2f}s — {len(boxes)} items (threshold={threshold:.2f})")

            with st.expander("Detailed results", expanded=True):
                st.markdown("""
                    <style>
                    .detection-table { width:100%; border-collapse: collapse; }
                    .detection-table th { background-color:#FF4B4B; color:white; padding:8px; text-align:left; }
                    .detection-table td { padding:8px; border-bottom:1px solid #ddd; }
                    </style>
                    <table class="detection-table">
                    <tr><th>#</th><th>Type</th><th>Confidence</th><th>Position</th></tr>
                """, unsafe_allow_html=True)

                rev_map = {1: "PCB", 2: "CAPACITOR", 3: "ALUMINIUM"}
                for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
                    b_arr = b.cpu().numpy() if hasattr(b, "cpu") else b
                    label_name = rev_map.get(l.item() if hasattr(l, "item") else l, "UNKNOWN")
                    st.markdown(f"<tr><td>{i+1}</td><td>{label_name}</td><td>{s:.2f}</td>"
                                f"<td>({b_arr[0]:.0f},{b_arr[1]:.0f}) → ({b_arr[2]:.0f},{b_arr[3]:.0f})</td></tr>",
                                unsafe_allow_html=True)
                st.markdown("</table>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
