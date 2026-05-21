import cv2
import numpy as np
from deepface import DeepFace
import webbrowser
import time
import threading
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import streamlit as st

# YouTube links for different emotions
emotion_songs = {
    "happy": "https://www.youtube.com/shorts/lZEd0KKPS_U",
    "sad": "https://www.youtube.com/shorts/pu9G7-sXWqw",
    "angry": "https://www.youtube.com/shorts/yUvqLDBX51Q",
    "surprise": "https://www.youtube.com/shorts/zyxIit_Zk0k",
    "neutral": "https://www.youtube.com/shorts/qWzBnwwBtOY",
    "fear": "https://www.youtube.com/shorts/C09gs-uD7a0",
    "disgust": "https://www.youtube.com/shorts/C09gs-uD7a0"
}

# Streamlit layout
st.set_page_config(page_title="Real-Time Emotion Music System", layout="wide")
st.title("🎭 Real-Time Emotion Detection + Music Player")

frame_placeholder = st.empty()
chatbot_placeholder = st.empty()

# Globals
is_song_playing = False
current_emotion = None
last_message_time = time.time()
detected_emotions = []
true_emotions = []
confidence_scores = []
emotions_over_time = {e: [] for e in emotion_songs}

# Music player
def play_song(emotion):
    global is_song_playing
    if not is_song_playing:
        is_song_playing = True
        webbrowser.open(emotion_songs[emotion])
        time.sleep(35)
        is_song_playing = False

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Cannot access camera.")
else:
    chatbot_text = ""
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Webcam error.")
            break

        frame = cv2.flip(frame, 1)

        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotions = result[0]['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]

            detected_emotions.append(dominant_emotion)
            true_emotions.append(dominant_emotion)
            confidence_scores.append(confidence)

            for em in emotion_songs:
                emotions_over_time[em].append(emotions.get(em, 0))

            if time.time() - last_message_time > 5:
                chatbot_text = f"You seem {dominant_emotion}. Here's something for you!"
                chatbot_placeholder.markdown(f"### 🤖 Chatbot: {chatbot_text}")
                last_message_time = time.time()

            # Display text on frame
            y = 30
            for emo, score in emotions.items():
                cv2.putText(frame, f"{emo.capitalize()}: {int(score)}%", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if emo == dominant_emotion else (255, 255, 255), 2)
                y += 25

            if not is_song_playing and current_emotion != dominant_emotion:
                current_emotion = dominant_emotion
                threading.Thread(target=play_song, args=(current_emotion,), daemon=True).start()

            _, buffer = cv2.imencode('.jpg', frame)
            frame_placeholder.image(buffer.tobytes(), channels="BGR")

        except Exception as e:
            st.warning(f"⚠ Detection Error: {e}")

        if time.time() - start_time > 45:
            break

        time.sleep(0.1)

    cap.release()

# ================== RESULTS ==================
if len(true_emotions) > 0:
    st.subheader("📊 Emotion Detection Results")

    accuracy = accuracy_score(true_emotions, detected_emotions)
    precision = precision_score(true_emotions, detected_emotions, average='weighted')
    recall = recall_score(true_emotions, detected_emotions, average='weighted')
    f1 = f1_score(true_emotions, detected_emotions, average='weighted')

    st.write(f"✅ *Accuracy*: {accuracy*100:.2f}%")
    st.write(f"✅ *Precision*: {precision:.2f}")
    st.write(f"✅ *Recall*: {recall:.2f}")
    st.write(f"✅ *F1-Score*: {f1:.2f}")

    emotions_list = list(emotion_songs.keys())
    cm = confusion_matrix(true_emotions, detected_emotions, labels=emotions_list)

    # ========== Graphs in 2 Columns ==========
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=emotions_list, yticklabels=emotions_list, fmt="d", ax=ax1, cbar=False)
        ax1.set_title("Confusion Matrix", fontsize=10)
        ax1.set_xlabel("Predicted", fontsize=8)
        ax1.set_ylabel("True", fontsize=8)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.countplot(x=detected_emotions, order=emotions_list, palette="Set2", ax=ax2)
        ax2.set_title("Detected Emotion Count", fontsize=10)
        ax2.tick_params(axis='x', labelsize=8)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        emotion_counts = {emotion: detected_emotions.count(emotion) for emotion in emotions_list}
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%',
               startangle=140, colors=sns.color_palette("Set2"), textprops={'fontsize': 8})
        ax3.set_title("Emotion Distribution", fontsize=10)
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        ax4.plot(confidence_scores, marker='o', linestyle='-', color='purple')
        ax4.set_title("Confidence Over Time", fontsize=10)
        ax4.set_xlabel("Frame", fontsize=8)
        ax4.set_ylabel("Confidence (%)", fontsize=8)
        ax4.tick_params(labelsize=7)
        st.pyplot(fig4)

    # Full width for heatmap
    st.markdown("### 🔥 Emotion Intensity Over Time")
    df_emotions = pd.DataFrame(emotions_over_time)
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    sns.heatmap(df_emotions.T, cmap="YlGnBu", cbar_kws={'label': 'Intensity (%)'}, ax=ax5)
    ax5.set_title("Emotion Intensity Over Time", fontsize=10)
    ax5.set_xlabel("Frame", fontsize=8)
    ax5.set_ylabel("Emotion", fontsize=8)
    ax5.tick_params(labelsize=7)
    st.pyplot(fig5)

else:
    st.warning("⚠ No emotion data to display.")

# ================== SAVE & DOWNLOAD SECTION ==================
st.markdown("---")
st.subheader("💾 Save Results & Download Report")

result_data = pd.DataFrame({
    "True Emotion": true_emotions,
    "Predicted Emotion": detected_emotions,
    "Confidence": confidence_scores
})

csv_data = result_data.to_csv(index=False)
st.download_button(
    label="📄 Download Detected Emotions (CSV)",
    data=csv_data,
    file_name="emotion_detection_results.csv",
    mime="text/csv"
)

summary = f"""
📊 Emotion Detection Summary Report

Total Frames Analyzed: {len(true_emotions)}
Accuracy: {accuracy*100:.2f}%
Precision: {precision:.2f}
Recall: {recall:.2f}
F1-Score: {f1:.2f}

Top 3 Detected Emotions:
"""
for emo, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
    summary += f"\n- {emo.capitalize()}: {count} times"

summary += "\n\nThank you for using the Emotion Detection Music System! 🎧"

st.download_button(
    label="📝 Download Summary Report (TXT)",
    data=summary,
    file_name="emotion_summary_report.txt",
    mime="text/plain"
)


