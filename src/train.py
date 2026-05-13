import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


def train_model():
    csv_path = "dataset_for_training.csv"
    if not os.path.exists(csv_path):
        print("❌ Hata: CSV dosyası bulunamadı!")
        return

    print("📊 1.5 Milyon veri yükleniyor, Monster PC'ye kuvvet... (Bu biraz sürebilir)")
    df = pd.read_csv(csv_path)

    # Özellikleri ve Hedefi ayır
    X = df[['width', 'length', 'height', 'distance']]
    y = df['label']

    print("🧠 Model eğitiliyor... (Random Forest ile yüksek hassasiyet hedefleniyor)")
    # n_jobs=-1 kullanarak işlemcinin tüm çekirdeklerini kullanıyoruz
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    # Eğitimi başlat
    model.fit(X, y)

    # Modeli Kaydet
    model_name = "odin_brain.pkl"
    joblib.dump(model, model_name)

    print("-" * 30)
    print(f"✅ EĞİTİM TAMAMLANDI!")
    print(f"📂 Model Dosyası: {model_name}")
    print("🚀 Artık sistemin 'Yapay Zeka' ile çalışmaya hazır!")
    print("-" * 30)


if __name__ == "__main__":
    train_model()