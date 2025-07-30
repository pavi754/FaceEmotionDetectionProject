import streamlit as st
import cv2
from deepface import DeepFace
from emotion_logger import log_emotion, init_csv
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

CSV_FILE = "emotion_data.csv"
PDF_FILE = "generated_report.pdf"

# Initialize CSV
init_csv()

# Page config
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("😄 Real-Time Face Emotion Detection")

# Controls
auto_clear = st.checkbox("🔄 Start fresh (clear old data)")
start_cam = st.button("📷 Start Webcam")
stop_cam = st.button("🛑 Stop Webcam")
clear_data = st.button("🗑️ Clear Data")

# Clear CSV manually
if clear_data:
    with open(CSV_FILE, "w") as f:
        f.write("timestamp,emotion\n")
    st.success("✅ Emotion data cleared!")

frame_display = st.image([])

# Webcam + Emotion Detection
if start_cam:
    if auto_clear:
        with open(CSV_FILE, "w") as f:
            f.write("timestamp,emotion\n")
        st.info("🧹 Old data cleared before starting webcam.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
    else:
        st.success("✅ Webcam started. Move into frame...")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read from webcam.")
                break

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    cv2.imwrite(tmp.name, frame)

                    result = DeepFace.analyze(img_path=tmp.name, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']

                    log_emotion(emotion)
                    cv2.putText(frame, f"{emotion}", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                st.warning(f"Detection error: {e}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(frame, channels="RGB")

            if stop_cam:
                cap.release()
                cv2.destroyAllWindows()
                st.success("🛑 Webcam stopped.")
                break

# 📊 Emotion Statistics
st.subheader("📊 Emotion Statistics")

if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    df = pd.read_csv(CSV_FILE)
    if not df.empty:
        emotion_counts = df["emotion"].value_counts()
        st.bar_chart(emotion_counts)

        st.markdown("#### 🥧 Pie Chart")
        fig1, ax1 = plt.subplots()
        ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis("equal")
        st.pyplot(fig1)

        st.markdown("#### 📈 Timeline")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        timeline = df.groupby(['timestamp', 'emotion']).size().unstack(fill_value=0)
        st.line_chart(timeline)

        with open(CSV_FILE, "rb") as f:
            st.download_button("⬇️ Download CSV", f, file_name="emotion_data.csv", mime="text/csv")
    else:
        st.info("CSV file is empty.")
else:
    st.info("CSV not found or no data yet.")

# 📄 Generate PDF Report
def generate_pdf():
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return False

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Emotion Detection Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
    pdf.ln(10)

    emotion_counts = df["emotion"].value_counts()
    for emotion, count in emotion_counts.items():
        pdf.cell(200, 10, txt=f"{emotion}: {count}", ln=1)

    pdf.output(PDF_FILE)
    return True

if st.button("📄 Generate PDF Report"):
    success = generate_pdf()
    if success:
        with open(PDF_FILE, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name="emotion_report.pdf", mime="application/pdf")
    else:
        st.warning("⚠️ No data to generate PDF.")
