
import time
import webbrowser
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace


SONG_LINKS = {
    "happy": "https://www.youtube.com/results?search_query=happy+feel+good+songs",
    "sad": "https://www.youtube.com/results?search_query=calm+comforting+songs",
    "angry": "https://www.youtube.com/results?search_query=relaxing+stress+relief+music",
    "surprise": "https://www.youtube.com/results?search_query=upbeat+surprise+playlist",
    "neutral": "https://www.youtube.com/results?search_query=lofi+focus+music",
    "fear": "https://www.youtube.com/results?search_query=calming+peaceful+music",
    "disgust": "https://www.youtube.com/results?search_query=positive+refreshing+music",
}

CHAT_MESSAGES = {
    "happy": "You look happy. I am suggesting an energetic playlist.",
    "sad": "You look a little sad. I am suggesting calm music.",
    "angry": "You look stressed or angry. I am suggesting relaxing music.",
    "surprise": "You look surprised. I am suggesting upbeat music.",
    "neutral": "You look neutral. I am suggesting focus music.",
    "fear": "You look worried. I am suggesting peaceful music.",
    "disgust": "You look uncomfortable. I am suggesting refreshing music.",
}

EMOTION_ORDER = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]
PREVIEW_WIDTH = 760


def normalize_deepface_result(result):
    """DeepFace may return either a dict or a list depending on version."""
    if isinstance(result, list):
        result = result[0]
    emotions = result.get("emotion", {})
    dominant = result.get("dominant_emotion") or max(emotions, key=emotions.get)
    confidence = float(emotions.get(dominant, 0.0))
    return dominant, confidence, emotions


def resize_for_preview(frame, width=PREVIEW_WIDTH):
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame

    ratio = width / current_width
    new_height = int(height * ratio)
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


def draw_overlay(frame, stable_emotion, confidence, emotions):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 92), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    title = f"Mood: {stable_emotion.upper()}  |  Confidence: {confidence:.1f}%"
    cv2.putText(frame, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    x = 14
    y = 58
    for emotion in EMOTION_ORDER:
        score = int(emotions.get(emotion, 0))
        color = (95, 220, 95) if emotion == stable_emotion else (170, 170, 170)
        cv2.putText(frame, f"{emotion}: {score}%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        x += 105
        if x > frame.shape[1] - 120:
            x = 14
            y += 22
    return frame


def recommendation_for(emotion):
    return {
        "emotion": emotion,
        "message": CHAT_MESSAGES.get(emotion, "I found your mood and selected music for you."),
        "song": SONG_LINKS.get(emotion, SONG_LINKS["neutral"]),
    }


def run_demo(duration, analysis_gap, smoothing_window, auto_open_music, music_cooldown):
    frame_box = st.empty()
    status_box = st.empty()
    recommendation_box = st.empty()
    progress_bar = st.progress(0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera not found. Please check webcam permission and try again.")
        return pd.DataFrame()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    history = []
    recent_predictions = deque(maxlen=smoothing_window)
    last_analysis = 0.0
    last_music_open = 0.0
    current_recommendation = None
    last_emotions = {emotion: 0.0 for emotion in EMOTION_ORDER}
    start_time = time.time()
    frame_number = 0

    try:
        while time.time() - start_time <= duration:
            ok, frame = cap.read()
            if not ok:
                st.warning("Could not read frame from webcam.")
                break

            frame_number += 1
            frame = cv2.flip(frame, 1)
            now = time.time()

            if now - last_analysis >= analysis_gap:
                last_analysis = now
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True,
                    )
                    predicted, confidence, emotions = normalize_deepface_result(result)
                    recent_predictions.append(predicted)
                    stable_emotion = Counter(recent_predictions).most_common(1)[0][0]
                    last_emotions = {emotion: float(emotions.get(emotion, 0.0)) for emotion in EMOTION_ORDER}

                    rec = recommendation_for(stable_emotion)
                    if current_recommendation != stable_emotion:
                        current_recommendation = stable_emotion
                        recommendation_box.info(f"{rec['message']}  Link: {rec['song']}")

                    if auto_open_music and now - last_music_open >= music_cooldown:
                        webbrowser.open(rec["song"])
                        last_music_open = now

                    history.append(
                        {
                            "time_seconds": round(now - start_time, 2),
                            "frame": frame_number,
                            "predicted_emotion": predicted,
                            "stable_emotion": stable_emotion,
                            "confidence_percent": round(confidence, 2),
                            **{f"{emotion}_score": round(last_emotions[emotion], 2) for emotion in EMOTION_ORDER},
                        }
                    )
                except Exception as exc:
                    status_box.warning(f"Emotion detection skipped for this frame: {exc}")

            stable = history[-1]["stable_emotion"] if history else "detecting"
            conf = history[-1]["confidence_percent"] if history else 0.0
            frame = resize_for_preview(frame)
            frame = draw_overlay(frame, stable, conf, last_emotions)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_box.image(frame_rgb, channels="RGB", width=PREVIEW_WIDTH)

            elapsed = time.time() - start_time
            progress_bar.progress(min(1.0, elapsed / duration))
            status_box.caption(f"Analyzing webcam... {int(max(0, duration - elapsed))} seconds left")
            time.sleep(0.03)
    finally:
        cap.release()

    return pd.DataFrame(history)


def show_results(results):
    if results.empty:
        st.warning("No emotion data was captured.")
        return

    st.subheader("Demo Results")

    top_emotion = results["stable_emotion"].mode().iloc[0]
    average_confidence = results["confidence_percent"].mean()
    stability = (results["stable_emotion"] == top_emotion).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Most detected mood", top_emotion.title())
    col2.metric("Average confidence", f"{average_confidence:.1f}%")
    col3.metric("Prediction stability", f"{stability:.1f}%")

    st.info(
        "Note: This live demo does not calculate real accuracy because there is no manually labelled ground truth. "
        "For an interview, explain confidence and stability here, and say that real accuracy needs a labelled test dataset."
    )

    counts = results["stable_emotion"].value_counts().reindex(EMOTION_ORDER, fill_value=0)
    st.bar_chart(counts)

    st.line_chart(results.set_index("time_seconds")["confidence_percent"])

    st.download_button(
        "Download CSV report",
        data=results.to_csv(index=False),
        file_name="emotion_music_demo_report.csv",
        mime="text/csv",
    )

    summary = (
        "Emotion Music Recommendation System - Demo Summary\n\n"
        f"Total analyzed frames: {len(results)}\n"
        f"Most detected mood: {top_emotion}\n"
        f"Average confidence: {average_confidence:.2f}%\n"
        f"Prediction stability: {stability:.2f}%\n\n"
        "Project idea: Detect the user's facial emotion from webcam video and recommend music "
        "that matches or improves the user's mood.\n"
    )
    st.download_button(
        "Download text summary",
        data=summary,
        file_name="emotion_music_summary.txt",
        mime="text/plain",
    )


def main():
    st.set_page_config(page_title="Emotion Music Recommender", layout="wide")

    st.title("Real-Time Emotion Based Music Recommender")
    st.caption("Webcam emotion detection using DeepFace, OpenCV, and Streamlit.")

    with st.sidebar:
        st.header("Demo Settings")
        duration = st.slider("Demo duration", 15, 90, 45, step=5)
        analysis_gap = st.slider("Analyze every seconds", 0.5, 3.0, 1.0, step=0.5)
        smoothing_window = st.slider("Smoothing window", 1, 7, 3, step=1)
        auto_open_music = st.checkbox("Automatically open music link", value=False)
        music_cooldown = st.slider("Music cooldown seconds", 10, 120, 35, step=5)

    intro_left, intro_right = st.columns([1.2, 0.8])
    with intro_left:
        st.markdown(
            """
            This project detects a face emotion from the webcam and recommends a playlist.
            It is designed for a short interview demo: start the camera, show the live mood,
            explain the recommendation, and download the report.
            """
        )
    with intro_right:
        st.markdown(
            """
            **Pipeline:** Camera -> OpenCV frame -> DeepFace emotion model -> smoothing ->
            music recommendation -> report.
            """
        )

    start = st.button("Start live demo", type="primary")
    if start:
        results = run_demo(duration, analysis_gap, smoothing_window, auto_open_music, music_cooldown)
        show_results(results)
    else:
        st.subheader("What this app can detect")
        preview = pd.DataFrame(
            {
                "emotion": EMOTION_ORDER,
                "recommendation_type": [
                    "energetic",
                    "calming",
                    "relaxing",
                    "upbeat",
                    "focus",
                    "peaceful",
                    "refreshing",
                ],
            }
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
