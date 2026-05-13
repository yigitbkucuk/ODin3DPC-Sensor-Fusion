import pandas as pd
import os

def check_data():
    csv_path = "dataset_for_training.csv"
    if not os.path.exists(csv_path):
        print("⏳ CSV henüz oluşmadı, hasatın ilk verileri toplamasını bekle...")
        return

    df = pd.read_csv(csv_path)
    print("--- 📊 MEVCUT VERİ DURUMU ---")
    print(f"Toplam Nesne Sayısı: {len(df)}")
    print("\nSınıf Dağılımı (0:Araç, 1:Yaya, 2:Obje):")
    print(df['label'].value_counts())
    print("----------------------------")

if __name__ == "__main__":
    check_data()