import open3d as o3d
import numpy as np
import os


def convert_to_bin(input_path, output_path):
    """
    iPhone/Polycam'den gelen .ply veya .pcd formatındaki nokta bulutunu,
    ODin3DPC'nin anlayacağı KITTI .bin formatına çevirir.
    """
    print(f">> Dosya okunuyor: {input_path}")

    # 1. Dosyayı Open3D ile oku
    pcd = o3d.io.read_point_cloud(input_path)

    if not pcd.has_points():
        print("HATA: Nokta bulutu boş veya dosya bozuk!")
        return False

    # 2. Noktaları (x, y, z) Numpy dizisine çevir (N, 3 formatında)
    points = np.asarray(pcd.points)

    # MÜHENDİSLİK DETAYI:
    # KITTI .bin formatı 4 boyutludur: (x, y, z, intensity).
    # iPhone lazeri "intensity" (yansıma şiddeti) verisini KITTI gibi ölçemez.
    # Bu yüzden sistemi kandırmamak için 4. boyutu (intensity) sıfırlarla dolduruyoruz.
    intensity = np.zeros((points.shape[0], 1))

    # 3. x, y, z ve intensity'yi yan yana birleştir (N, 4 formatında)
    kitti_data = np.hstack((points, intensity)).astype(np.float32)

    # 4. Saf binary (.bin) olarak kaydet
    kitti_data.tofile(output_path)

    print(f">> Başarı! Veri KITTI formatına çevrildi: {output_path}")
    print(f">> İşlenen Toplam Nokta Sayısı: {points.shape[0]}")
    return True


# --- TOPLU DÖNÜŞTÜRÜCÜ (KLASÖR İÇİN) ---
def convert_folder(input_folder, output_folder):
    """Bir klasör dolusu iPhone verisini tek tıkla çevirir."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_exts = ['.ply', '.pcd']
    files = [f for f in os.listdir(input_folder) if any(f.endswith(ext) for ext in valid_exts)]

    if not files:
        print(f"Uyarı: {input_folder} içinde çevrilecek .ply veya .pcd dosyası bulunamadı.")
        return

    for i, file_name in enumerate(sorted(files)):
        in_path = os.path.join(input_folder, file_name)
        # 000001.bin, 000002.bin şeklinde sırayla isimlendir
        out_name = f"{str(i).zfill(6)}.bin"
        out_path = os.path.join(output_folder, out_name)

        convert_to_bin(in_path, out_path)


if __name__ == "__main__":
    # Test için klasör yolları (iPhone verileri geldiğinde burayı kullanacağız)
    RAW_IPHONE_DIR = "../raw_iphone_data"
    READY_DATA_DIR = "../data/iphone_converted"

    # Klasör yoksa oluşturalım ki hata vermesin
    if not os.path.exists(RAW_IPHONE_DIR):
        os.makedirs(RAW_IPHONE_DIR)
        print(f">> '{RAW_IPHONE_DIR}' klasörü oluşturuldu. iPhone verilerini buraya at kanka.")
    else:
        print(">> Dönüştürme işlemi başlıyor...")
        convert_folder(RAW_IPHONE_DIR, READY_DATA_DIR)