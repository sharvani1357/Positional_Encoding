import streamlit as st
import pandas as pd
import numpy as np
import re

from PyPDF2 import PdfReader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# -----------------------------------
# TRAIN MODEL
# -----------------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv("Resume.csv")

    resume_col = None

    for col in df.columns:
        if "resume" in col.lower():
            resume_col = col
            break

    def clean_text(text):

        text = str(text).lower()

        text = re.sub(
            r'[^a-zA-Z\s]',
            ' ',
            text
        )

        text = re.sub(
            r'\s+',
            ' ',
            text
        )

        return text

    df["clean_resume"] = df[
        resume_col
    ].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=5000
    )

    X = vectorizer.fit_transform(
        df["clean_resume"]
    )

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df["Category"]
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = model.score(
        X_test,
        y_test
    )

    return (
        model,
        vectorizer,
        encoder,
        accuracy
    )

model, vectorizer, encoder, accuracy = (
    train_model()
)

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown("""
<div class="main-title">
🚀 AI Resume Analyzer
</div>

<div class="sub-title">
Resume Classification using NLP
</div>
""",
unsafe_allow_html=True)

# -----------------------------------
# DASHBOARD
# -----------------------------------

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown(
    f"""
    <div class="metric-card">
    <h3>Dataset Accuracy</h3>
    <h2>{accuracy*100:.2f}%</h2>
    </div>
    """,
    unsafe_allow_html=True
    )

with c2:
    st.markdown(
    """
    <div class="metric-card">
    <h3>Algorithm</h3>
    <h2>TF-IDF + LR</h2>
    </div>
    """,
    unsafe_allow_html=True
    )

with c3:
    st.markdown(
    """
    <div class="metric-card">
    <h3>Deployment</h3>
    <h2>Streamlit</h2>
    </div>
    """,
    unsafe_allow_html=True
    )

st.write("")

# -----------------------------------
# PDF UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

resume_text = ""

if uploaded_file:

    pdf = PdfReader(
        uploaded_file
    )

    for page in pdf.pages:

        text = page.extract_text()

        if text:
            resume_text += text

resume_input = st.text_area(
    "Or Paste Resume Text",
    value=resume_text,
    height=300
)

# -----------------------------------
# SKILLS
# -----------------------------------

skills_db = [

    "python",
    "java",
    "sql",
    "mysql",
    "tensorflow",
    "keras",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "html",
    "css",
    "javascript",
    "react",
    "c++"
]

def extract_skills(text):

    found = []

    text = text.lower()

    for skill in skills_db:

        if skill in text:
            found.append(skill)

    return list(set(found))

# -----------------------------------
# EXPERIENCE
# -----------------------------------

def get_experience(text):

    matches = re.findall(
        r'(\d+)\s+years',
        text.lower()
    )

    if not matches:
        return "Fresher"

    years = max(
        [int(x) for x in matches]
    )

    if years <= 1:
        return "Fresher"

    elif years <= 4:
        return "Junior"

    elif years <= 8:
        return "Mid-Level"

    else:
        return "Senior"

# -----------------------------------
# PREDICTION
# -----------------------------------

def predict_category(text):

    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text.lower()
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    vector = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        vector
    )[0]

    probability = np.max(
        model.predict_proba(
            vector
        )
    )

    category = encoder.inverse_transform(
        [prediction]
    )[0]

    return (
        category,
        probability*100
    )

# -----------------------------------
# BUTTON
# -----------------------------------

if st.button(
    "Analyze Resume",
    use_container_width=True
):

    if len(
        resume_input.strip()
    ) == 0:

        st.warning(
            "Please enter resume."
        )

    else:

        category, confidence = (
            predict_category(
                resume_input
            )
        )

        skills = extract_skills(
            resume_input
        )

        exp = get_experience(
            resume_input
        )

        st.write("")

        st.markdown(
        f"""
        <div class='result-card'>
        🎯 Predicted Category
        <br><br>
        {category}
        </div>
        """,
        unsafe_allow_html=True
        )

        st.write("")

        a,b = st.columns(2)

        with a:
            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

        with b:
            st.metric(
                "Experience Level",
                exp
            )

        st.write("")

        st.subheader(
            "🛠 Extracted Skills"
        )

        if skills:

            html = ""

            for skill in skills:

                html += (
                    f"<span class='skill'>"
                    f"{skill}"
                    f"</span>"
                )

            st.markdown(
                html,
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No skills detected."
            )

st.write("---")

st.markdown(
"""
<div class='footer'>
AI Resume Classification System
<br>
Built using NLP, TF-IDF and Machine Learning
</div>
""",
unsafe_allow_html=True
)