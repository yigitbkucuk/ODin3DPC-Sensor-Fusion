import os
import sys

# Proje ana dizinini Python yoluna ekle (Import hatalarını kökten çözer)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Processor'dan sadece ihtiyacımız olanı alıyoruz
try:
    from src.processor import process_frame
except ImportError as e:
    print(f"❌ Hata: 'src.processor' bulunamadı: {e}")
    sys.exit(1)


def start_harvesting():
    # Veri yolunu proje köküne göre ayarla
    velo_path = os.path.join(project_root, "data", "velodyne")

    if not os.path.exists(velo_path):
        print(f"❌ Hata: '{velo_path}' klasörü bulunamadı!")
        return

    bin_files = sorted([f for f in os.listdir(velo_path) if f.endswith('.bin')])
    total_files = len(bin_files)

    if total_files == 0:
        print("⚠️ Klasörde .bin dosyası yok!")
        return

    print(f"🚀 Hasat Başlıyor! {total_files} dosya taranacak...")
    print("-" * 50)

    for i, bin_file in enumerate(bin_files):
        full_path = os.path.join(velo_path, bin_file)
        try:
            # Bu fonksiyon zaten kendi içinde save_training_feature'ı çağırıyor
            process_frame(full_path)

            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                print(f"📦 İlerleme: {i + 1}/{total_files} [%{int((i + 1) / total_files * 100)}]")
        except Exception as e:
            print(f"⚠️ {bin_file} hatası: {e}")

    print("-" * 50)
    print("✅ Hasat Tamamlandı! dataset_for_training.csv oluştu.")


if __name__ == "__main__":
    start_harvesting()