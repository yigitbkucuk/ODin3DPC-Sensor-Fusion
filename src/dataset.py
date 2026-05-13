import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


class ODin3DDataset(Dataset):
    def __init__(self, bin_dir, csv_path):
        self.bin_dir = bin_dir
        # Rapor dosyasındaki verileri yükle
        self.labels_df = pd.read_csv(csv_path)
        self.bin_files = sorted(os.listdir(bin_dir))

    def __len__(self):
        return len(self.bin_files)

    def __getitem__(self, idx):
        # 1. LiDAR verisini oku (.bin -> Tensor)
        bin_file = self.bin_files[idx]
        bin_path = os.path.join(self.bin_dir, bin_file)
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        # Sadece x,y,z koordinatlarını al ve Tensor'a çevir
        point_tensor = torch.from_numpy(points[:, :3]).float()

        # 2. CSV'den bu kareye ait etiketleri (labels) çek
        frame_labels = self.labels_df[self.labels_df['Kare_Adi'] == bin_file]

        # Basitlik için sadece nesne sayısını ve mesafeyi hedef olarak veriyoruz
        # H200'de bu kısım tam 3D Bounding Box koordinatlarına (x,y,z,w,l,h) dönecek
        target = {
            "object_count": torch.tensor(len(frame_labels)),
            "distances": torch.tensor(frame_labels['Mesafe_Metre'].values).float()
        }

        return point_tensor, target