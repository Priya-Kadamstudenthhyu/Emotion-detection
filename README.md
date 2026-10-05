# Emotion Detection and Music Recommendation

A Python project that uses a webcam to estimate facial emotions and suggest music. It includes a Streamlit demo for an interactive presentation and a small OpenCV program for a direct webcam preview.

## Features

- Captures webcam frames with OpenCV.
- Uses DeepFace to estimate scores for seven emotions: happy, sad, angry, surprise, neutral, fear, and disgust.
- Displays emotion predictions and confidence estimates.
- Maps detected emotions to music search links and recommendation messages.
- The Streamlit demo can smooth recent predictions, display summary charts, and download CSV and text reports.

## Project files

- `app.py` — configurable Streamlit demo with a live preview, prediction smoothing, music recommendations, charts, and downloadable reports. This is the recommended entry point.
- `act.py` — alternate Streamlit implementation with a fixed-length detection session and additional charts.
- `emotion_realtime.py` — basic OpenCV webcam window that overlays the dominant emotion. Press **Q** to quit.
- `requirements.txt` — Python package dependencies.
- `true_labels.csv` — sample emotion labels. The current scripts do not load this file for evaluation.

## Requirements

- Python 3.10 or newer is recommended.
- A webcam and permission for Python to access it.
- Internet access may be needed the first time DeepFace downloads its model files.

## Setup

Open a terminal in this project directory and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For `act.py`, install its additional imports:

```powershell
python -m pip install seaborn scikit-learn
```

## Run the project

Start the recommended Streamlit app:

```powershell
streamlit run app.py
```

Streamlit prints a local URL to open in a browser. Select **Start live demo** and allow webcam access. The sidebar controls the demo length, analysis interval, prediction smoothing, and optional automatic opening of music links.

Run the alternate Streamlit app with:

```powershell
streamlit run act.py
```

Run the basic OpenCV webcam program with:

```powershell
python emotion_realtime.py
```

## How it works

1. OpenCV captures frames from the default camera (camera index `0`).
2. DeepFace analyzes frames and returns emotion scores and a dominant emotion.
3. In `app.py`, recent dominant predictions are smoothed with a configurable majority window to reduce rapid changes in the displayed mood.
4. The app maps the selected emotion to a recommendation message and a YouTube search link.
5. At the end of the demo, the app summarizes the captured predictions and offers downloadable reports.

## Notes on model results

The displayed confidence is the model's score for a prediction; it is not a validated accuracy measurement. The live demo does not have independently labeled ground-truth frames, so it cannot calculate meaningful accuracy. `true_labels.csv` is not currently connected to the evaluation code. In `act.py`, the current `true_emotions` values are copied from predictions, so its accuracy, precision, recall, and F1 values should not be treated as real evaluation results.

Facial emotion predictions are estimates and can vary with lighting, camera quality, face position, and other conditions. This project is a demonstration, not a clinical or psychological assessment tool.

## Troubleshooting

- **Camera not found:** Check camera permissions and close other apps that may be using the webcam. The scripts use camera index `0` by default.
- **PowerShell blocks activation:** Run `python -m pip install -r requirements.txt` using the environment's Python executable, or activate the environment from Command Prompt with `.venv\Scripts\activate.bat`.
- **Missing module:** Ensure the virtual environment is active and install the package named in the error. `act.py` additionally needs `seaborn` and `scikit-learn`.
