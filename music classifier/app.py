from flask import Flask, render_template, request
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("genre_model.h5")
classes = np.load("genre_labels.npy")

os.makedirs("static", exist_ok=True)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled.reshape(1, -1)
    except Exception as e:
        print("Feature extraction error:", e)
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    file_url = None

    if request.method == "POST":
        if "file" not in request.files:
            prediction = "No file part"
        else:
            file = request.files["file"]
            if file.filename == "":
                prediction = "No selected file"
            else:
                
                filename = file.filename
                if not filename.endswith(".wav"):
                    filename += ".wav"
                filepath = os.path.join("static", filename)
                file.save(filepath)
                file_url = filepath

                
                features = extract_features(filepath)
                if features is not None:
                    preds = model.predict(features)
                    pred_genre = classes[np.argmax(preds)]
                    prediction = f"Predicted Genre: {pred_genre}"
                else:
                    prediction = "Could not extract features."

    return render_template("index.html", prediction=prediction, file=file_url)

if __name__ == "__main__":
    app.run(debug=True)
