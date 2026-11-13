import streamlit as st
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
import json
import html
import re
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PlantLens AI",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# GEMINI CONNECTION
# ============================================================

@st.cache_resource
def init_gemini():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )


gemini_client = init_gemini()


# ============================================================
# SESSION STATE
# ============================================================

if "plant_doctor_messages" not in st.session_state:
    st.session_state.plant_doctor_messages = []

user_name = "Plant Explorer"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "./models/plant_disease_resnet18_best.pth"
DISEASE_PATH = None

HIGH_CONFIDENCE = 70
MEDIUM_CONFIDENCE = 40


# ============================================================
# CUSTOM CSS — NATURE / GREEN THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #F4FAF3;
}

.block-container {
    padding-top: 3.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* HERO */

.hero-title {
    font-size: 58px;
    font-weight: 800;
    color: #20352A;
    margin-bottom: 5px;
    letter-spacing: -2px;
}

.hero-title span {
    color: #3D8B5F;
}

.hero-subtitle {
    font-size: 22px;
    color: #617468;
    margin-bottom: 30px;
}

.hero-card {
    background: #E5F2E7;
    border-radius: 28px;
    padding: 45px;
    margin-bottom: 30px;
    border: 1px solid #CDE3D0;
}


/* SECTIONS */

.section-title {
    font-size: 30px;
    font-weight: 750;
    color: #20352A;
    margin-bottom: 8px;
}

.section-text {
    color: #65766A;
    font-size: 16px;
    margin-bottom: 20px;
}


/* FEATURE CARDS */

.feature-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    min-height: 165px;
    box-sizing: border-box;
    border: 1px solid #D9E8DC;
    box-shadow: 0 5px 20px rgba(35,80,45,.07);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: floatingCard 4s ease-in-out infinite;
}

.feature-card-1 {
    animation-delay: 0s;
}

.feature-card-2 {
    animation-delay: .8s;
}

.feature-card-3 {
    animation-delay: 1.6s;
}

@keyframes floatingCard {
    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-7px);
    }
}

.feature-card:hover {
    box-shadow: 0 14px 32px rgba(35,80,45,.13);
    border-color: #B9D5BE;
}

.feature-icon {
    font-size: 30px;
    line-height: 1;
    height: 34px;
    display: flex;
    align-items: center;
    margin-bottom: 13px;
}

.feature-title {
    font-size: 17px;
    font-weight: 700;
    color: #20352A;
    line-height: 1.3;
    margin-bottom: 7px;
}

.feature-text {
    font-size: 13px;
    color: #65766A;
    line-height: 1.5;
    max-width: 260px;
}


/* BUTTONS */

.stButton > button {
    background: #3D8B5F;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 16px;
    width: 100%;
}

.stButton > button:hover {
    background: #32754F;
    color: white;
}


/* RESULT CARDS */

.result-card,
.disease-card {
    background: white;
    padding: 30px;
    border-radius: 22px;
    border: 1px solid #D9E8DC;
    box-shadow: 0 8px 30px rgba(35,80,45,.08);
    margin-top: 10px;
}

.result-label {
    font-size: 17px;
    color: #789080;
    margin-bottom: 8px;
}

.result-disease {
    font-size: 32px;
    font-weight: 800;
    color: #20352A;
    margin-bottom: 12px;
    text-transform: capitalize;
}

.confidence-label {
    font-size: 15px;
    color: #65766A;
    margin-bottom: 6px;
}

.confidence-value {
    font-size: 24px;
    font-weight: 800;
    color: #3D8B5F;
}


/* DISEASE INFORMATION */

.disease-title {
    font-size: 25px;
    font-weight: 800;
    color: #20352A;
    margin-bottom: 24px;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 26px;
}

.info-box {
    background: #EAF5EC;
    border-radius: 16px;
    padding: 17px 10px;
    text-align: center;
    border: 1px solid #CDE3D0;
}

.info-number {
    font-size: 18px;
    font-weight: 800;
    color: #3D8B5F;
}

.info-label {
    font-size: 12px;
    color: #65766A;
    margin-top: 4px;
}

.disease-heading {
    font-size: 18px;
    font-weight: 750;
    color: #20352A;
    margin-top: 20px;
    margin-bottom: 10px;
}

.disease-item {
    color: #526158;
    padding: 5px 0;
    line-height: 1.5;
    font-size: 15px;
}


/* CONFIDENCE WARNING */

.low-confidence-card {
    background: #FFF7E8;
    border: 1px solid #EBCF91;
    border-radius: 18px;
    padding: 18px 22px;
    margin-top: 18px;
}

.low-confidence-title {
    font-size: 16px;
    font-weight: 750;
    color: #8A641E;
    margin-bottom: 6px;
}

.low-confidence-text {
    font-size: 14px;
    color: #6F6145;
    line-height: 1.6;
}

.possible-note {
    background: #F5FAF5;
    border: 1px solid #D9E8DC;
    border-radius: 14px;
    padding: 13px 16px;
    margin-bottom: 15px;
    color: #65766A;
    font-size: 14px;
    line-height: 1.5;
}


/* EXPANDER */

div[data-testid="stExpander"] {
    background: #fff !important;
    border: 2px solid #B9D5BE !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 20px rgba(35,80,45,.08) !important;
    margin-bottom: 18px !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] summary {
    background: #EAF5EC !important;
    color: #20352A !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    font-size: 16px !important;
    font-weight: 750 !important;
    min-height: 55px !important;
}

div[data-testid="stExpander"] summary:hover {
    background: #DDEEE0 !important;
}

div[data-testid="stExpander"] summary span {
    color: #20352A !important;
    font-weight: 750 !important;
}

div[data-testid="stExpander"] summary svg {
    color: #3D8B5F !important;
    fill: #3D8B5F !important;
}


/* PLANT DOCTOR */

.st-key-plant_doctor_chat_box {
    background: #fff !important;
    border: 1px solid #CDE3D0 !important;
    border-radius: 28px !important;
    padding: 32px 34px 28px !important;
    box-shadow: 0 12px 38px rgba(35,80,45,.10) !important;
    animation: floatingPlantChat 4.5s ease-in-out infinite;
    margin-top: 8px !important;
    margin-bottom: 28px !important;
}

@keyframes floatingPlantChat {
    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-7px);
    }
}

.plant-doctor-title {
    font-size: 30px;
    font-weight: 800;
    color: #20352A !important;
    margin-bottom: 5px;
}

.plant-doctor-subtitle {
    font-size: 15px;
    color: #617468 !important;
    line-height: 1.55;
}

.plant-doctor-ready {
    background: #EAF5EC !important;
    border: 1px solid #CDE3D0 !important;
    border-radius: 17px !important;
    padding: 15px 18px !important;
    margin: 20px 0 22px !important;
    color: #526158 !important;
    line-height: 1.55 !important;
}

.plant-doctor-info {
    background: #F5FAF5 !important;
    border: 1px solid #D9E8DC !important;
    border-radius: 17px !important;
    padding: 15px 18px !important;
    margin: 20px 0 22px !important;
    color: #65766A !important;
    line-height: 1.55 !important;
}

.plant-doctor-user-message {
    padding: 13px 16px;
    margin: 8px 0 10px;
    color: #20352A !important;
    font-size: 15px;
    line-height: 1.55;
    font-weight: 600;
}

.plant-doctor-user-label,
.plant-doctor-ai-label {
    color: #3D8B5F !important;
    font-weight: 800;
}

.plant-doctor-ai-message {
    padding: 14px 17px 16px;
    margin-bottom: 12px;
    background: #F4FAF3;
    border-radius: 16px;
    color: #20352A !important;
    font-size: 15px;
    line-height: 1.65;
    border: 1px solid #D9E8DC;
}

.plant-doctor-ai-label {
    display: block;
    margin-bottom: 7px;
}

.plant-doctor-ai-content,
.plant-doctor-ai-content p,
.plant-doctor-ai-content li {
    color: #20352A !important;
}

.plant-doctor-ai-content p {
    margin: 5px 0 9px;
}

.plant-doctor-ai-content ul,
.plant-doctor-ai-content ol {
    margin: 6px 0 9px;
    padding-left: 25px;
}

.plant-doctor-ai-content li {
    margin-bottom: 5px;
}

.plant-doctor-input-label {
    color: #526158 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    margin: 8px 0 !important;
}

.plant-doctor-input-hint {
    color: #789080 !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    margin: -2px 0 10px !important;
}

.st-key-plant_doctor_input input {
    background: #fff !important;
    color: #20352A !important;
    border: 1px solid #CDE3D0 !important;
    border-radius: 15px !important;
    min-height: 48px !important;
    font-size: 15px !important;
}

.st-key-plant_doctor_input input::placeholder {
    color: #718176 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

.st-key-plant_doctor_input input:focus {
    border-color: #3D8B5F !important;
    box-shadow: 0 0 0 1px #3D8B5F !important;
}

.st-key-clear_plant_doctor button {
    background: #F5FAF5 !important;
    color: #65766A !important;
    border: 1px solid #CDE3D0 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}


/* ALERTS */

div[data-testid="stAlert"] {
    background: #FFFBE5 !important;
    border: 1px solid #E9DEA8 !important;
    border-radius: 12px !important;
    color: #5E563E !important;
}

div[data-testid="stAlert"] * {
    color: #5E563E !important;
    opacity: 1 !important;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #789080;
    margin-top: 50px;
    font-size: 14px;
}


/* MOBILE */

@media(max-width:768px) {

    .block-container {
        padding-top: 2rem;
    }

    .hero-title {
        font-size: 42px;
    }

    .hero-subtitle {
        font-size: 19px;
    }

    .section-title {
        font-size: 25px;
    }

    .feature-card {
        min-height: 145px;
    }

    .info-grid {
        grid-template-columns: 1fr;
    }

    .st-key-plant_doctor_chat_box {
        padding: 25px 20px 22px !important;
        border-radius: 23px !important;
    }

    .plant-doctor-title {
        font-size: 26px;
    }

    .plant-doctor-ai-message,
    .plant-doctor-user-message,
    .plant-doctor-ai-content {
        font-size: 14px;
    }

    .hero-card {
        padding: 30px 25px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_plant_model():

    device = torch.device("cpu")

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, CLASS_NAMES, device


# ============================================================
# DISEASE DATABASE
# ============================================================

@st.cache_data
def load_diseases():
    return {}


def is_healthy_prediction(name):
    return "healthy" in str(name).lower()

def format_disease_name(name):
    return (
        str(name)
        .replace("___", " - ")
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )

def predict_disease(image, model, class_names, device):
    image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    return predicted_class, confidence_percent

def get_matching_disease(
    predicted,
    diseases
):

    target = (
        str(predicted)
        .lower()
        .strip()
    )

    for item in diseases:

        name = str(
            item.get(
                "disease",
                item.get("name", "")
            )
        ).lower().strip()

        if name == target:
            return item

    return None


def as_list(value):

    if not value:
        return []

    if isinstance(value, list):
        return value

    return [value]


# ============================================================
# DISPLAY DISEASE INFORMATION
# ============================================================

def display_disease_info(disease):

    name = html.escape(
        str(
            disease.get(
                "disease",
                disease.get(
                    "name",
                    "Unknown disease"
                )
            )
        )
    )

    plant = html.escape(
        str(
            disease.get(
                "plant",
                "Not specified"
            )
        )
    )

    severity = html.escape(
        str(
            disease.get(
                "severity",
                "Not specified"
            )
        )
    )

    cause = html.escape(
        str(
            disease.get(
                "cause",
                "Not specified"
            )
        )
    )

    symptoms = as_list(
        disease.get("symptoms")
    )

    treatment = as_list(
        disease.get(
            "treatment",
            disease.get("treatments")
        )
    )

    prevention = as_list(
        disease.get(
            "prevention",
            disease.get("preventions")
        )
    )

    def items(values):

        if not values:
            return (
                '<div class="disease-item">'
                'No information available.'
                '</div>'
            )

        return "".join(
            f'<div class="disease-item">'
            f'• {html.escape(str(x))}'
            f'</div>'
            for x in values
        )

    st.html(
        f"""
        <div class="disease-card">

            <div class="disease-title">
                🌿 {name}
            </div>

            <div class="info-grid">

                <div class="info-box">
                    <div class="info-number">
                        {plant}
                    </div>
                    <div class="info-label">
                        Plant
                    </div>
                </div>

                <div class="info-box">
                    <div class="info-number">
                        {severity}
                    </div>
                    <div class="info-label">
                        Severity
                    </div>
                </div>

                <div class="info-box">
                    <div class="info-number">
                        {cause}
                    </div>
                    <div class="info-label">
                        Common Cause
                    </div>
                </div>

            </div>

            <div class="disease-heading">
                🔎 Symptoms
            </div>

            {items(symptoms)}

            <div class="disease-heading">
                💊 Recommended Treatment
            </div>

            {items(treatment)}

            <div class="disease-heading">
                🛡️ Prevention
            </div>

            {items(prevention)}

        </div>
        """
    )


# ============================================================
# PLANT DOCTOR — GEMINI
# ============================================================

def ask_plant_doctor(
    question,
    predicted_disease=None,
    disease_info=None
):

    disease_context = (
        format_disease_name(predicted_disease)
        if predicted_disease
        else "No disease has been identified yet."
    )

    database_context = (
        "No disease database information is selected."
    )

    if disease_info:

        database_context = f"""
Disease:
{disease_info.get(
    'disease',
    disease_info.get('name', 'Unknown')
)}

Plant:
{disease_info.get('plant', 'Unknown')}

Severity:
{disease_info.get('severity', 'Unknown')}

Cause:
{disease_info.get('cause', 'Unknown')}

Symptoms:
{disease_info.get('symptoms', [])}

Treatment:
{disease_info.get(
    'treatment',
    disease_info.get('treatments', [])
)}

Prevention:
{disease_info.get(
    'prevention',
    disease_info.get('preventions', [])
)}
"""

    history = "\n".join(
        (
            "User"
            if m["role"] == "user"
            else "Plant Doctor"
        )
        + f": {m['content']}"
        for m in st.session_state
        .plant_doctor_messages[-6:]
    )

    if not history:
        history = "No previous conversation."

    system_instruction = """
You are Plant Doctor, a friendly AI assistant
specialized in plant health, gardening and plant care.

Answer naturally, practically and clearly.

You can help with:

- Plant diseases
- Symptoms
- Treatment
- Prevention
- Pests
- Watering
- Sunlight
- Soil
- Fertilizer
- Pruning
- Propagation
- Houseplants
- Vegetables
- Flowers
- Leaf yellowing
- Leaf spots
- Wilting
- Curling
- Repotting
- Seasonal care
- General plant science

An image is optional.

Use detected disease context when relevant,
but do not force it into unrelated questions.

If detection is uncertain, describe it as a
possible prediction rather than a confirmed diagnosis.

Do not invent information as if it came from
the PlantLens database.

For unclear or serious plant problems, recommend
checking the whole plant or consulting a qualified
local agriculture or horticulture professional.

If the question is completely unrelated to plants,
politely explain that Plant Doctor focuses on plants,
gardening and plant health.

Use conversation history for follow-up questions.

Keep answers practical and easy to read.
"""

    prompt = f"""
Detected disease:
{disease_context}

Disease database context:
{database_context}

Recent conversation:
{history}

User's latest question:
{question}

Answer the latest question as Plant Doctor.
"""

    for attempt in range(2):

        try:

            response = gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.4,
                    max_output_tokens=500,
                    candidate_count=1
                )
            )

            if response and response.text:
                return response.text.strip()

        except Exception:

            if attempt == 0:
                continue

    raise Exception(
        "Plant Doctor is temporarily unavailable. "
        "Please try again in a few seconds."
    )


# ============================================================
# FORMAT AI RESPONSE
# ============================================================

def format_ai_response(text):

    text = html.escape(str(text))

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        text
    )

    text = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"<em>\1</em>",
        text
    )

    text = re.sub(
        r"`([^`]+)`",
        r"<code>\1</code>",
        text
    )

    lines = text.split("\n")
    output = []

    in_ul = False
    in_ol = False

    for line in lines:

        s = line.strip()

        if not s:

            if in_ul:
                output.append("</ul>")
                in_ul = False

            if in_ol:
                output.append("</ol>")
                in_ol = False

            continue

        bullet = re.match(
            r"^[-•]\s+(.+)",
            s
        )

        number = re.match(
            r"^\d+\.\s+(.+)",
            s
        )

        if bullet:

            if in_ol:
                output.append("</ol>")
                in_ol = False

            if not in_ul:
                output.append("<ul>")
                in_ul = True

            output.append(
                f"<li>{bullet.group(1)}</li>"
            )

            continue

        if number:

            if in_ul:
                output.append("</ul>")
                in_ul = False

            if not in_ol:
                output.append("<ol>")
                in_ol = True

            output.append(
                f"<li>{number.group(1)}</li>"
            )

            continue

        if in_ul:
            output.append("</ul>")
            in_ul = False

        if in_ol:
            output.append("</ol>")
            in_ol = False

        output.append(
            f"<p>{s}</p>"
        )

    if in_ul:
        output.append("</ul>")

    if in_ol:
        output.append("</ol>")

    return "".join(output)


# ============================================================
# GREETING + HERO
# ============================================================

hour = datetime.now().hour

if hour < 12:

    greeting = "Good morning"
    icon = "☀️"

elif hour < 17:

    greeting = "Good afternoon"
    icon = "🌤️"

else:

    greeting = "Good evening"
    icon = "🌙"


st.html(
    f"""
    <div class="hero-card">

        <div class="hero-title">
            🌿 PlantLens <span>AI</span>
        </div>

        <div style="
            font-size:28px;
            font-weight:800;
            color:#20352A;
            margin-top:10px;
            margin-bottom:8px;
        ">
            {greeting}, {html.escape(user_name)}! {icon}
        </div>

        <div class="hero-subtitle">
            See it. Know it. Grow it.
        </div>

        <div style="
            font-size:17px;
            color:#526158;
            max-width:650px;
            line-height:1.7;
        ">
            Welcome to your intelligent plant health space.
            Upload a leaf or plant photo and let PlantLens AI
            help identify possible diseases and guide you
            toward better care.
        </div>

    </div>
    """
)


# ============================================================
# UPLOAD + ANALYSIS
# ============================================================

st.html(
    '<div class="section-title">'
    '🌱 What is happening to your plant?'
    '</div>'
)

st.html(
    '<div class="section-text">'
    'Upload a clear leaf or plant photo and let '
    'PlantLens AI do the detective work.'
    '</div>'
)


uploaded_file = st.file_uploader(
    "Upload plant image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    st.markdown("### ✨ Your plant")

    col1, col2 = st.columns([1.1, 1])

    with col1:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            width=500
        )

    with col2:

        st.html(
            """
            <div class="result-card">

                <div class="result-label">
                    Ready for analysis
                </div>

                <div style="
                    font-size:30px;
                    font-weight:800;
                    color:#20352A;
                    margin-top:8px;
                ">
                    🔍 Detect plant disease
                </div>

                <p style="
                    color:#65766A;
                    line-height:1.6;
                ">
                    PlantLens AI will analyze your image
                    using our trained ResNet18
                    deep-learning model.
                </p>

            </div>
            """
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        if st.button("🔍 Analyze Plant"):

            with st.spinner(
                "🧠 PlantLens AI is analyzing your plant..."
            ):

                try:

                    model, classes, device = (
                        load_plant_model()
                    )

                    predicted, confidence = (
                        predict_disease(
                            image,
                            model,
                            classes,
                            device
                        )
                    )

                    diseases = load_diseases()

                    disease_info = (
                        get_matching_disease(
                            predicted,
                            diseases
                        )
                    )

                    st.session_state[
                        "predicted_disease"
                    ] = predicted

                    st.session_state[
                        "confidence"
                    ] = confidence

                    st.session_state[
                        "disease_info"
                    ] = disease_info

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Something went wrong while "
                        f"analyzing the image: {e}"
                    )


# ============================================================
# DISPLAY RESULT + TREATMENT
# ============================================================

if "predicted_disease" in st.session_state:

    predicted = (
        st.session_state[
            "predicted_disease"
        ]
    )

    confidence = (
        st.session_state[
            "confidence"
        ]
    )

    disease_info = (
        st.session_state.get(
            "disease_info"
        )
    )

    display_name = format_disease_name(
        predicted
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # ========================================================
    # HEALTHY / DISEASE CONFIDENCE MESSAGE
    # ========================================================

    if is_healthy_prediction(predicted):

        if confidence >= HIGH_CONFIDENCE:

            st.success(
                "Healthy plant detected with high confidence! 🌱"
            )

            label = "🌱 PlantLens AI identified"

        elif confidence >= MEDIUM_CONFIDENCE:

            st.info(
                "The plant appears healthy, "
                "but the model has moderate confidence. 🌱"
            )

            label = "🌱 PlantLens AI's likely match"

        else:

            st.warning(
                "The model is uncertain. "
                "Please inspect the plant carefully. ⚠️"
            )

            label = "⚠️ PlantLens AI's uncertain match"

    elif confidence >= HIGH_CONFIDENCE:

        st.success(
            "Plant disease identified with high confidence! 🦠"
        )

        label = "🧠 PlantLens AI identified"

    elif confidence >= MEDIUM_CONFIDENCE:

        st.warning(
            "PlantLens AI found a possible disease match. "
            "The prediction may not be exact. ⚠️"
        )

        label = "🧠 PlantLens AI's possible match"

    else:

        label = "🧠 PlantLens AI's best guess"


    # ========================================================
    # RESULT CARD
    # ========================================================

    st.html(
        f"""
        <div class="result-card">

            <div class="result-label">
                {html.escape(label)}
            </div>

            <div class="result-disease">
                {html.escape(display_name)}
            </div>

            <div class="confidence-label">
                Model confidence
            </div>

            <div class="confidence-value">
                {confidence:.2f}%
            </div>

        </div>
        """
    )


    # ========================================================
    # LOW CONFIDENCE WARNING
    # ========================================================

    if confidence < MEDIUM_CONFIDENCE:

        st.html(
            f"""
            <div class="low-confidence-card">

                <div class="low-confidence-title">
                    ⚠️ PlantLens AI couldn't confidently
                    identify this prediction.
                </div>

                <div class="low-confidence-text">

                    <b>
                        {html.escape(display_name)}
                    </b>

                    is the model's best guess, but
                    confidence is only

                    <b>
                        {confidence:.2f}%
                    </b>.

                    Try a clearer image showing the
                    affected leaf area.

                </div>

            </div>
            """
        )


    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    # Healthy predictions do NOT show disease information.

    if (
        not is_healthy_prediction(predicted)
        and confidence >= HIGH_CONFIDENCE
    ):

        st.html(
            '<div class="section-title">'
            '🌿 Disease Information & Treatment'
            '</div>'
        )

        st.html(
            '<div class="section-text">'
            'Helpful information based on the disease '
            'PlantLens AI identified.'
            '</div>'
        )

        if disease_info:

            display_disease_info(
                disease_info
            )

        else:

            st.info(
                f"We don't have treatment information "
                f"for '{display_name}' yet. "
                "More plant-care information is "
                "coming soon! 🌱"
            )


    elif (
        not is_healthy_prediction(predicted)
        and confidence >= MEDIUM_CONFIDENCE
    ):

        st.html(
            '<div class="section-title">'
            '🌿 Possible Disease Information'
            '</div>'
        )

        st.html(
            f"""
            <div class="section-text">
                This information is based on PlantLens AI's
                possible match:
                <b>{html.escape(display_name)}</b>.
                Please verify the prediction before treatment.
            </div>
            """
        )

        if disease_info:

            display_disease_info(
                disease_info
            )

        else:

            st.info(
                f"We don't have treatment information "
                f"for '{display_name}' yet."
            )


    elif not is_healthy_prediction(predicted):

        st.html(
            '<div class="section-title">'
            '🌿 Possible Disease'
            '</div>'
        )

        st.html(
            '<div class="section-text">'
            'Because the prediction is uncertain, '
            'PlantLens AI will not present it as a '
            'confirmed diagnosis.'
            '</div>'
        )

        if disease_info:

            disease_name = html.escape(
                str(
                    disease_info.get(
                        "disease",
                        disease_info.get(
                            "name",
                            display_name
                        )
                    )
                )
            )

            with st.expander(
                f"🌿 View possible treatment: "
                f"{disease_name}"
            ):

                st.html(
                    """
                    <div class="possible-note">
                        ⚠️ This information is based on
                        the model's uncertain prediction.
                        Please verify the plant problem
                        before starting treatment.
                    </div>
                    """
                )

                display_disease_info(
                    disease_info
                )


# ============================================================
# FEATURES
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)

st.html(
    '<div class="section-title">'
    '🌿 From leaf to healthy plant'
    '</div>'
)

st.html(
    '<div class="section-text">'
    'Everything you need to understand plant problems '
    'and improve your plant care.'
    '</div>'
)


col1, col2, col3 = st.columns(3)


with col1:

    st.html(
        """
        <div class="feature-card feature-card-1">

            <div class="feature-icon">
                🧠
            </div>

            <div class="feature-title">
                AI Disease Recognition
            </div>

            <div class="feature-text">
                Identify possible plant diseases using
                our trained deep-learning model.
            </div>

        </div>
        """
    )


with col2:

    st.html(
        """
        <div class="feature-card feature-card-2">

            <div class="feature-icon">
                🌱
            </div>

            <div class="feature-title">
                Treatment Guidance
            </div>

            <div class="feature-text">
                Understand symptoms, treatment options
                and prevention steps.
            </div>

        </div>
        """
    )


with col3:

    st.html(
        """
        <div class="feature-card feature-card-3">

            <div class="feature-icon">
                👨‍🌾
            </div>

            <div class="feature-title">
                Plant Doctor
            </div>

            <div class="feature-text">
                Get help with plant health, pests,
                gardening and everyday plant care.
            </div>

        </div>
        """
    )


# ============================================================
# PLANT DOCTOR CHAT
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)

ai_disease = (
    st.session_state.get(
        "predicted_disease"
    )
)

ai_disease_info = (
    st.session_state.get(
        "disease_info"
    )
)


with st.container(
    border=True,
    key="plant_doctor_chat_box"
):

    st.html(
        """
        <div class="plant-doctor-title">
            👨‍🌾 Plant Doctor
        </div>

        <div class="plant-doctor-subtitle">
            Ask your Plant Doctor anything about plant
            health, diseases, gardening, pests, soil,
            watering or care.
        </div>
        """
    )


    if ai_disease:

        st.html(
            f"""
            <div class="plant-doctor-ready">

                🌿 <strong>Plant Doctor is ready</strong>

                <br>

                Your detected disease:

                <strong>
                    {html.escape(
                        format_disease_name(
                            ai_disease
                        )
                    )}
                </strong>

            </div>
            """
        )

    else:

        st.html(
            """
            <div class="plant-doctor-info">

                💡 No image is required.

                Ask Plant Doctor anything about plant
                diseases, gardening, watering, soil,
                pests, fertilizer or plant care.

            </div>
            """
        )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if st.session_state.plant_doctor_messages:

        for message in (
            st.session_state
            .plant_doctor_messages
        ):

            content = html.escape(
                str(message["content"])
            )

            if message["role"] == "user":

                st.html(
                    f"""
                    <div class="plant-doctor-user-message">

                        <span class="
                            plant-doctor-user-label
                        ">
                            👤 You:
                        </span>

                        {content.replace(
                            chr(10),
                            "<br>"
                        )}

                    </div>
                    """
                )

            else:

                st.html(
                    f"""
                    <div class="
                        plant-doctor-ai-message
                    ">

                        <span class="
                            plant-doctor-ai-label
                        ">
                            🤖 Plant Doctor
                        </span>

                        <div class="
                            plant-doctor-ai-content
                        ">
                            {format_ai_response(
                                message["content"]
                            )}
                        </div>

                    </div>
                    """
                )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    st.html(
        """
        <div class="plant-doctor-input-label">
            Ask your Plant Doctor anything 🌱
        </div>

        <div class="plant-doctor-input-hint">
            Try: “Why are my leaves turning yellow?”
            · “How often should I water my tomato plant?”
            · “How can I treat fungal disease?”
        </div>
        """
    )


    with st.form(
        "plant_doctor_form",
        clear_on_submit=True,
        border=False
    ):

        input_col, send_col = st.columns(
            [5, 1],
            vertical_alignment="bottom"
        )

        with input_col:

            question = st.text_input(
                "Ask Plant Doctor something...",
                placeholder=(
                    "Ask anything about plants, "
                    "diseases, gardening or care..."
                ),
                label_visibility="collapsed",
                key="plant_doctor_input"
            )

        with send_col:

            send = st.form_submit_button(
                "Send",
                key="plant_doctor_send",
                width="stretch"
            )


    # ========================================================
    # SEND MESSAGE
    # ========================================================

    if send:

        clean = question.strip()

        if not clean:

            st.warning(
                "Please enter a question "
                "for Plant Doctor."
            )

        else:

            st.session_state.plant_doctor_messages.append(
                {
                    "role": "user",
                    "content": clean
                }
            )

            with st.spinner(
                "👨‍🌾 Plant Doctor is thinking..."
            ):

                try:

                    answer = ask_plant_doctor(
                        clean,
                        ai_disease,
                        ai_disease_info
                    )

                    st.session_state.plant_doctor_messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    st.rerun()

                except Exception:

                    st.session_state.plant_doctor_messages.pop()

                    st.error(
                        "❌ Plant Doctor could not "
                        "respond right now."
                    )

                    st.caption(
                        "Please try again in a few seconds."
                    )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.session_state.plant_doctor_messages:

        if st.button(
            "🗑️ Clear Plant Doctor Chat",
            key="clear_plant_doctor",
            width="stretch"
        ):

            st.session_state.plant_doctor_messages = []

            st.rerun()


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)

st.html(
    '<div class="section-title">'
    '⚙️ How PlantLens AI works'
    '</div>'
)

st.html(
    """
    <div style="
        background:#20352A;
        color:white;
        border-radius:24px;
        padding:35px;
        margin-top:15px;
    ">

        <div style="
            display:flex;
            justify-content:space-between;
            text-align:center;
        ">

            <div>
                <div style="font-size:35px;">
                    📸
                </div>

                <b>1. Upload</b>

                <p style="color:#CFE0D3;">
                    Choose a plant image
                </p>
            </div>


            <div style="font-size:30px;">
                →
            </div>


            <div>
                <div style="font-size:35px;">
                    🧠
                </div>

                <b>2. Analyze</b>

                <p style="color:#CFE0D3;">
                    AI detects a possible disease
                </p>
            </div>


            <div style="font-size:30px;">
                →
            </div>


            <div>
                <div style="font-size:35px;">
                    🌿
                </div>

                <b>3. Understand</b>

                <p style="color:#CFE0D3;">
                    Explore symptoms & treatment
                </p>
            </div>


            <div style="font-size:30px;">
                →
            </div>


            <div>
                <div style="font-size:35px;">
                    👨‍🌾
                </div>

                <b>4. Care</b>

                <p style="color:#CFE0D3;">
                    Improve plant health
                </p>
            </div>

        </div>

    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">
        🌿 PlantLens AI
        &nbsp;•&nbsp;
        See it. Know it. Grow it.
    </div>
    """
)










