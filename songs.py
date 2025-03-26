import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "C:\\Users\\prasa\\Downloads\\large_songs_dataset.csv"

try:
    with open(file_path, "r", encoding="utf-8") as file:
        df = pd.read_csv(file)
    
    print("Dataset loaded successfully!")
    print("Columns in dataset:", df.columns.tolist())

    if "mood" not in df.columns:
        raise ValueError("The dataset does not contain a 'mood' column. Please check the dataset.")

    features = ["danceability", "energy", "valence", "tempo", "acousticness", "liveness", "speechiness"]
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[features]
    y = df["mood"]

    if df[features + ["mood"]].isnull().any().sum() > 0:
        print("Warning: Dataset contains missing values. Filling with mean values.")
        df = df.fillna(df.mean())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.countplot(x=df["mood"], order=df["mood"].value_counts().index, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Mood Distribution in Spotify Songs")
    plt.xlabel("Mood")
    plt.ylabel("Count")
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Make sure the file is in the correct location.")
except PermissionError:
    print(f"Error: Permission denied for '{file_path}'. Try running as administrator.")
except ValueError as ve:
    print(f"Dataset Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
