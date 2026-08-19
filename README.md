# English Accent Classifier

A small Streamlit demonstration that classifies a WAV speech recording as Indian, British, or American English. It uses a classical pattern-recognition pipeline rather than a neural network.

## Architecture

```text
WAV audio → 16 kHz mono audio → 13 MFCCs → mean + standard deviation
→ 26 features → StandardScaler → RBF SVM → Accent prediction
```

`audio_features.py` owns the feature extraction used by both training and the app. `train.py` fits and saves one scikit-learn pipeline containing both the `StandardScaler` and the RBF `SVC`; this prevents training and inference preprocessing from drifting apart.

## Install

```bash
pip install -r requirements.txt
```

## Dataset layout

Download/place the AccentDataset locally so it contains these directories (subdirectories are supported):

```text
AccentDataset/
├── indian_accent/
├── British_accent/
└── American_accent/
```

Empty, corrupt, and unreadable WAV files are skipped during training.

## Use an existing model

If you already have a trained `accent_model.pkl`, copy it into this project directory beside `app.py`. The Streamlit app uses that file automatically. It also accepts a file named `model.pkl` for compatibility.

## Train (optional)

From this project directory, run:

```bash
python train.py /path/to/AccentDataset
```

This writes `model.pkl` alongside the scripts by default. To choose another destination:

```bash
python train.py /path/to/AccentDataset --output model.pkl
```

The script prints the number of usable training samples and discovered classes.

## Run the app

With `accent_model.pkl` (or `model.pkl`) beside `app.py`, start the app with:

```bash
streamlit run app.py
```

Upload a `.wav` speech clip, listen to the playback, then select **Predict Accent**. The app shows the most likely accent and the probability for each class.

## Limitations

This is a student-project prototype trained on a small dataset. Its results can be sensitive to microphone quality, recording conditions, dataset bias, and the speaker's individual background. It recognizes only the three training categories and should not be used to make decisions about people.
