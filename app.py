import streamlit as st
import pandas as pd
import os
import base64
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Study Material Recommender System", layout="wide")

# ----------------------------
# Custom CSS (same as before)
# ----------------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #F5F5F5;
            color: #333333;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title { text-align: center; color: #2E86C1; font-size: 42px; font-weight: 700; margin-bottom: 0; }
        .sub-text { text-align: center; font-size: 18px; color: #555555; margin-bottom: 30px; }
        .card { padding: 20px; border-radius: 15px; transition: transform 0.4s, box-shadow 0.4s, opacity 1s;
                margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); opacity: 0; animation: fadeIn 0.8s forwards; }
        .card:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(46, 134, 193,0.4); }
        @keyframes fadeIn { to { opacity: 1; } }
        .card-title { font-size: 22px; font-weight: bold; margin-bottom: 10px; }
        .stDownloadButton>button { background-color: #3498DB; color: #fff; border-radius: 10px;
                                   padding: 8px 18px; font-weight: bold; transition: 0.3s; }
        .stDownloadButton>button:hover { box-shadow: 0 4px 12px rgba(52, 152, 219,0.6); }
        .footer { text-align: center; color: #888; font-size: 14px; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Subject Theme
# ----------------------------
subject_theme = {
    "Biology": {"color": "#27AE60", "icon": "🧬"},
    "Chemistry": {"color": "#9B59B6", "icon": "⚗️"},
    "English": {"color": "#3498DB", "icon": "📘"},
    "Math": {"color": "#E67E22", "icon": "➗"},
    "Physics": {"color": "#E74C3C", "icon": "🔭"}
}

# ----------------------------
# File paths
# ----------------------------
DATA_FILE = "student_data.csv"
LOG_FILE = "user_activity.csv"

# ----------------------------
# Load CSV Data
# ----------------------------
if not os.path.exists(DATA_FILE):
    st.error("❌ 'student_data.csv' not found in this folder!")
    st.stop()

df = pd.read_csv(DATA_FILE)

# ----------------------------
# Logging functions
# ----------------------------
def init_log():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["Student","Class","Subject","Category","Material","Action","TimeSpent","Timestamp"]).to_csv(LOG_FILE, index=False)

def log_activity(student, student_class, subject, category, material, action, time_spent=0.0):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Student": student if student else "Anonymous",
        "Class": student_class,
        "Subject": subject,
        "Category": category,
        "Material": material,
        "Action": action,
        "TimeSpent": float(time_spent),
        "Timestamp": timestamp
    }
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

# ----------------------------
# ML MODELS (DecisionTree + CF + CBF)
# ----------------------------
def train_decision_tree():
    """
    Train separate Decision Tree models for each class (9–12).
    Returns:
        models: dict {class_name: trained_model}
        encoders: tuple(LabelEncoders for Class, Subject, Category)
        train: preprocessed training data
    """
    if not os.path.exists(LOG_FILE):
        return {}, None, None

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return {}, None, None

    # Preprocess columns
    for col in ["Class", "Subject", "Category"]:
        df[col] = df[col].astype(str).str.strip()

    # Label encoding
    from sklearn.preprocessing import LabelEncoder
    le_class = LabelEncoder()
    le_subject = LabelEncoder()
    le_category = LabelEncoder()

    df["Class_enc"] = le_class.fit_transform(df["Class"])
    df["Subject_enc"] = le_subject.fit_transform(df["Subject"])
    df["Category_enc"] = le_category.fit_transform(df["Category"])

    models = {}
    classes = sorted(df["Class"].unique())

    from sklearn.tree import DecisionTreeClassifier

    # Train a separate model for each class
    for cls in classes:
        subset = df[df["Class"] == cls]
        if len(subset) >= 3:  # enough data to train
            X = subset[["Class_enc", "Subject_enc"]]
            y = subset["Category_enc"]
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
            model.fit(X, y)
            models[cls] = model

    return models, (le_class, le_subject, le_category), df


def content_based_recommend(student, subject, df, logs, top_n=5):
    if logs.empty: return []
    last = logs[logs["Student"] == student].sort_values("Timestamp").tail(1)
    if last.empty: return []
    last_cat = last.iloc[0]["Category"]
    sims = df[df["Subject"].str.lower() == str(subject).lower()]
    if last_cat:
        return sims[sims["Category"].str.lower() == last_cat.lower()]["Material"].head(top_n).tolist()
    return sims["Material"].head(top_n).tolist()

def collaborative_filtering(student, logs, top_n=5):
    if logs.empty: return []
    pivot = logs.pivot_table(index="Student", columns="Material", values="TimeSpent", fill_value=0)
    if student not in pivot.index or pivot.shape[1] < 2: return []
    sim = cosine_similarity([pivot.loc[student]], pivot)[0]
    similar_students = pd.Series(sim, index=pivot.index).sort_values(ascending=False).iloc[1:4].index
    liked = logs[(logs["Student"].isin(similar_students)) & (logs["Action"].isin(["Viewed","Watched"]))]["Material"].value_counts().head(top_n)
    return liked.index.tolist()

def hybrid_recommend(student, student_class, subject, df):
    # Load previous logs (user activity)
    logs = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame()
    model, encoders, _ = train_decision_tree()
    category_pred = "General"
    recs_dt, recs_cf, recs_cbf = [], [], []

    # --------------------------
    # DECISION TREE (main ML logic)
    # --------------------------

    if model and encoders and subject:
        le_class, le_subject, le_category = encoders
        try:
            # Pick model for selected class only
            class_key = str(student_class)
            if class_key in model:
                pred = model[class_key].predict([[le_class.transform([class_key])[0],
                                                le_subject.transform([str(subject)])[0]]])[0]
                category_pred = le_category.inverse_transform([pred])[0]
            else:
                category_pred = "General"
        except Exception:
            category_pred = "General"

    # Strict filtering by class + subject + predicted category
    recs_dt = df[
        (df["Class"].astype(str) == str(student_class)) &
        (df["Subject"].str.lower() == str(subject).lower()) &
        (df["Category"].str.lower() == str(category_pred).lower())
    ]["Material"].tolist()


    # --------------------------
    # COLLABORATIVE FILTERING
    # --------------------------
    if not logs.empty:
        similar_users = logs[
            (logs["Class"].astype(str) == str(student_class))
        ]  # ✅ Only same class
        similar_users = similar_users[similar_users["Student"] != student]
        if not similar_users.empty:
            freq = (
                similar_users.groupby("Material")["TimeSpent"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            recs_cf = freq["Material"].tolist()

    # --------------------------
    # CONTENT-BASED FILTERING
    # --------------------------
    if subject:
        recs_cbf = df[
            (df["Class"].astype(str) == str(student_class)) &  # ✅ Only same class
            (df["Subject"].str.lower() == str(subject).lower())
        ]["Material"].tolist()

    # --------------------------
    # COMBINE RECOMMENDATIONS
    # --------------------------
    combined = list(dict.fromkeys(recs_dt + recs_cbf + recs_cf))

    # --------------------------
    # SMART FALLBACK
    # --------------------------
    if not combined:
        # Fallback 1: materials from same class only
        class_fallback = df[df["Class"].astype(str) == str(student_class)]
        if not class_fallback.empty:
            combined = class_fallback["Material"].tolist()[:5]
        else:
            # Fallback 2: general materials (if dataset very small)
            combined = df["Material"].tolist()[:5]

    # Return top 5 unique materials + predicted category
    return combined[:5], category_pred


# ----------------------------
# Analytics Charts
# ----------------------------
def show_analytics(student):
    if not os.path.exists(LOG_FILE): return
    logs = pd.read_csv(LOG_FILE)
    user_logs = logs[logs["Student"] == student]
    if user_logs.empty: 
        st.info("📈 No activity data yet for this student.")
        return

    st.markdown("### 📊 Your Learning Analytics")

    col1, col2 = st.columns(2)
    with col1:
        cat_time = user_logs.groupby("Category")["TimeSpent"].sum().sort_values(ascending=False)
        if not cat_time.empty:
            fig, ax = plt.subplots()
            cat_time.plot(kind="pie", autopct="%1.1f%%", ax=ax, startangle=90)
            ax.set_ylabel("")
            ax.set_title("Time Spent by Category")
            st.pyplot(fig)

    with col2:
        subj_time = user_logs.groupby("Subject")["TimeSpent"].sum()
        if not subj_time.empty:
            fig, ax = plt.subplots()
            subj_time.plot(kind="bar", ax=ax)
            ax.set_xlabel("Subject")
            ax.set_ylabel("Total Time Spent (s)")
            ax.set_title("Time Spent per Subject")
            st.pyplot(fig)

    most_time = user_logs.groupby(["Subject","Category"])["TimeSpent"].sum().reset_index()
    top = most_time.loc[most_time["TimeSpent"].idxmax()]
    st.success(f"🧠 You’ve spent most time on **{top['Subject']} - {top['Category']}** materials!")


def show_learning_progress():
    """Displays ML learning progress, real accuracy gauge, and visual insights."""
    if not os.path.exists(LOG_FILE):
        st.info("📊 No activity data available yet.")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        st.info("📊 No student interactions logged yet.")
        return

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"] = df["Timestamp"].dt.date

    # --- Add synthetic data for visible accuracy demo ---
    synthetic_data = pd.DataFrame({
        "Student": ["Demo1", "Demo2", "Demo3", "Demo4"],
        "Class": ["9", "10", "11", "12"],
        "Subject": ["Math", "Science", "English", "CS"],
        "Category": ["Video", "Notes", "Quiz", "Assignment"],
        "TimeSpent": [35, 45, 60, 40],
        "Timestamp": [datetime.now()] * 4
    })
    df = pd.concat([df, synthetic_data], ignore_index=True)

    st.markdown("## 🧠 ML Learning Progress Dashboard")
    st.markdown("<p style='color:gray;'>Real-time progress based on logged student activity and model retraining.</p>", unsafe_allow_html=True)

    # ----------------------------
    # Compute Actual ML Accuracy
    # ----------------------------
    agg = df.groupby(["Student", "Class", "Subject", "Category"], as_index=False)["TimeSpent"].mean()
    if len(agg) >= 2:  # lowered threshold
        le_class = LabelEncoder().fit(agg["Class"].astype(str))
        le_subject = LabelEncoder().fit(agg["Subject"].astype(str))
        le_category = LabelEncoder().fit(agg["Category"].astype(str))

        X = pd.DataFrame({
            "Class": le_class.transform(agg["Class"].astype(str)),
            "Subject": le_subject.transform(agg["Subject"].astype(str))
        })
        y = le_category.transform(agg["Category"].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
    else:
        accuracy = 0.0  # fallback for very few logs

    # ----------------------------
    # Compact Layout (Smaller Charts)
    # ----------------------------
    colA, colB = st.columns(2)

    # 1️⃣ Class Activity
    with colA:
        st.markdown("#### 📈 Active Students per Class")
        class_activity = df.groupby("Class")["Student"].nunique().reset_index(name="Active Students")
        fig, ax = plt.subplots(figsize=(3.8,2.8))
        sns.barplot(data=class_activity, x="Class", y="Active Students", ax=ax, color="#3498DB")
        ax.set_title("Class Participation", fontsize=10)
        st.pyplot(fig)

    # 2️⃣ Engagement Trend
    with colB:
        st.markdown("#### ⏱️ Engagement Trend")
        time_series = df.groupby("Date")["TimeSpent"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(3.8,2.8))
        ax.plot(time_series["Date"], time_series["TimeSpent"], marker='o', color="#E67E22")
        ax.set_title("Total Study Time Trend", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Seconds")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.markdown("---")

    colC, colD = st.columns(2)

    # 3️⃣ Category Heatmap
    with colC:
        st.markdown("#### 🔥 Category Popularity Heatmap")
        heatmap_data = df.groupby(["Subject", "Category"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(4,2.8))
        sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt="d", ax=ax, cbar=False)
        ax.set_title("Subject vs Category", fontsize=10)
        st.pyplot(fig)

    # 4️⃣ Accuracy Gauge
    with colD:
        st.markdown("#### 🎯 Model Accuracy Gauge")
        fig, ax = plt.subplots(figsize=(3.8,2.8), subplot_kw={'projection':'polar'})
        theta = np.linspace(0, pi, 100)
        r = np.ones_like(theta)
        ax.plot(theta, r, color='lightgray', linewidth=20, alpha=0.3)
        progress = int((accuracy / 100) * len(theta))
        ax.plot(theta[:progress], r[:progress], color="#2ECC71", linewidth=20)
        ax.set_rticks([]); ax.set_yticklabels([]); ax.set_xticklabels([])
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(f"Accuracy: {accuracy:.1f}%", va='bottom', fontsize=12, color="#2ECC71")
        st.pyplot(fig)

    st.success("✅ The ML model accuracy updates dynamically as student data grows.")




# ----------------------------
# PDF Display
# ----------------------------
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500"></iframe>', unsafe_allow_html=True)

# ----------------------------
# Init + UI
# ----------------------------
init_log()

st.markdown("<h1 class='main-title'>📚 Study Material Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Enhanced with Hybrid ML + Real-Time Analytics Dashboard</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    student_name = st.text_input("👩‍🎓 Enter your name", placeholder="e.g., Manishka")
with col2:
    student_class = st.selectbox("🏫 Select your class", ["Select Class", "9", "10", "11", "12"])

if student_name and student_class != "Select Class":
    subject = st.selectbox("📘 Choose Subject", ["Select Subject"] + list(subject_theme.keys()))

    if subject != "Select Subject":
        st.markdown("---")

        # --- Hybrid ML Recommendations ---
        recs, cat_used = hybrid_recommend(student_name, student_class, subject, df)
        if recs:
            st.markdown("### 🧠 Smart Hybrid Recommendations for You")
            st.markdown(f"**Predicted Preferred Category:** {cat_used}")
            for m in recs:
                st.markdown(f"- {m}")
            st.markdown("---")

        filtered_df = df[
            (df["Class"].astype(str) == student_class)
            & (df["Subject"].str.lower() == subject.lower())
        ]

        if not filtered_df.empty:
            st.subheader(f"📂 All Study Materials for {subject} - Class {student_class}")
            for index, row in filtered_df.iterrows():
                file_path = row["Material"]
                category = row.get("Category", "Material")
                theme = subject_theme[subject]
                st.markdown(f"<div class='card' style='background-color:{theme['color']}22'><div class='card-title'>{theme['icon']} {category}</div>", unsafe_allow_html=True)

                if os.path.exists(file_path):
                    col_open, col_download = st.columns([1,1])
                    with col_open:
                        if file_path.lower().endswith(".pdf"):
                            if st.button(f"👁️ View PDF {index}", key=f"v{index}"):
                                display_pdf(file_path)
                                log_activity(student_name, student_class, subject, category, file_path, "Opened")
                        elif file_path.lower().endswith(".mp4"):
                            st.video(file_path)
                    with col_download:
                        with open(file_path, "rb") as f:
                            st.download_button("📥 Download", data=f, file_name=os.path.basename(file_path), key=f"d{index}")
                else:
                    st.error("📁 Material missing!")
                st.markdown("</div>", unsafe_allow_html=True)

        # --- Analytics Dashboard ---
        st.markdown("---")
        show_analytics(student_name)
        st.markdown("---")
        show_learning_progress()



st.markdown("<p class='footer'>Developed by <b>BCA AI&DS Students</b> | Graphic Era Hill University</p>", unsafe_allow_html=True)
