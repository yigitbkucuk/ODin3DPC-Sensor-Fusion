import open3d as o3d
import numpy as np
from ultralytics import YOLO
import cv2
import os
import joblib
import pandas as pd
import warnings

# Gereksiz kütüphane uyarılarını gizle
warnings.filterwarnings("ignore", category=UserWarning)

# --- DİNAMİK DOSYA YOLU TANIMLAMALARI ---
# Bu dosyanın (processor.py) bulunduğu klasörü (src) al
current_dir = os.path.dirname(os.path.abspath(__file__))

# YOLO modelini yükle (best.pt artık src klasörünün içinde olduğu için direkt current_dir kullanıyoruz)
yolo_model_path = os.path.join(current_dir, "best.pt")
model_2d = YOLO(yolo_model_path)

# Yapay Zeka Modelini (odin_brain.pkl) proje ana dizininden yükle (src'nin bir üst klasörü)
project_root = os.path.dirname(current_dir)
ml_model_path = os.path.join(project_root, "odin_brain.pkl")

try:
    odin_model = joblib.load(ml_model_path)
    print("Yapay Zeka Modeli (odin_brain) basariyla yuklendi")
except Exception as e:
    odin_model = None
    print(f"UYARI: odin_brain.pkl yuklenemedi, sadece temel boyut analizi calisacak. Hata: {e}")

# Hız tahmini için nesne hafızası (ID tabanlı takip)
prev_objs = {}


def load_kitti_bin(bin_path):
    """LiDAR .bin dosyasını okur ve Open3D PointCloud objesine çevirir."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    # Bellek çökmesini önlemek için ardışık (contiguous) hale getiriyoruz
    points = np.ascontiguousarray(scan[:, 0:3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def read_kitti_calib(calib_path):
    """KITTI kalibrasyon dosyasını okur."""
    if not os.path.exists(calib_path): return None, None
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    Tr, R0 = None, None
    for line in lines:
        if line.startswith('Tr_velo_to_cam'):
            Tr = np.array([float(v) for v in line.strip().split()[1:]]).reshape(3, 4)
        elif line.startswith('R0_rect'):
            R0 = np.array([float(v) for v in line.strip().split()[1:]]).reshape(3, 3)
    return Tr, R0


def transform_point_cloud(pcd, Tr, R0):
    """LiDAR verisini kamera koordinat sistemine taşır."""
    if Tr is None: return pcd
    points = np.asarray(pcd.points)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = (Tr @ points_h.T).T
    if R0 is not None:
        transformed = (R0 @ transformed.T).T

    transformed = np.ascontiguousarray(transformed)
    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(transformed)
    return pcd_transformed


def process_frame(bin_path):
    """Ana LiDAR işleme motoru: Segmentasyon, Kümeleme ve AI Sınıflandırma."""
    global prev_objs
    calib_path = bin_path.replace('velodyne', 'calib').replace('.bin', '.txt')
    Tr, R0 = read_kitti_calib(calib_path)
    pcd = load_kitti_bin(bin_path)

    # Koordinat dönüşümü (Kamera Perspektifi için kritik)
    if Tr is not None:
        pcd = transform_point_cloud(pcd, Tr, R0)

    # 1. Zemin Ayıklama (RANSAC Plane Segmentation)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.38, ransac_n=3, num_iterations=100)
    objects = pcd.select_by_index(inliers, invert=True)

    # 2. Nesne Kümeleme (DBSCAN)
    labels = np.array(objects.cluster_dbscan(eps=0.75, min_points=6, print_progress=False))

    bboxes, metadata = [], []

    if labels.size > 0 and labels.max() >= 0:
        for i in range(labels.max() + 1):
            indices = np.where(labels == i)[0]
            if len(indices) < 6: continue

            cluster = objects.select_by_index(indices)
            bbox = cluster.get_axis_aligned_bounding_box()
            w, l, h = bbox.get_extent()
            center = bbox.get_center()
            dist = np.linalg.norm(center)

            label_text = "Engel"
            bbox.color = [0.5, 0.5, 0.5]  # Varsayılan gri renk

            # --- GENİŞLETİLMİŞ YAPAY ZEKA KARAR MANTIKLARI ---
            if odin_model:
                feat = pd.DataFrame([[w, l, h, dist]], columns=['width', 'length', 'height', 'distance'])
                pred = odin_model.predict(feat)[0]

                if pred == 0:  # Arac Kategorisi
                    if l > 9.0 or h > 3.0:
                        label_text = "Otobus / Tir"
                        bbox.color = [0.8, 0.1, 0.1]
                    elif l > 6.0 or h > 2.2:
                        label_text = "Kamyon"
                        bbox.color = [1.0, 0.3, 0.1]
                    elif l < 2.5 and w < 1.2:
                        label_text = "Motosiklet"
                        bbox.color = [1.0, 0.5, 0.0]
                    else:
                        label_text = "Binek Arac"
                        bbox.color = [1, 0, 0]

                elif pred == 1:  # Yaya Kategorisi
                    if h < 1.3:
                        label_text = "Cocuk / Hayvan"
                        bbox.color = [0.2, 0.8, 0.2]
                    elif l > 1.2 and w < 1.0:
                        label_text = "Bisikletli"
                        bbox.color = [0.0, 1.0, 0.5]
                    else:
                        label_text = "Yaya"
                        bbox.color = [0, 1, 0]

                else:  # Cevresel Kategorisi
                    if h > 3.5 and l > 4.0:
                        label_text = "Bina"
                        bbox.color = [0.4, 0.4, 0.4]
                    elif 0.5 < w < 2.0 and h > 1.8:
                        label_text = "Agac"
                        bbox.color = [0.2, 0.6, 0.2]
                    elif w < 1.0 and l < 1.0 and h > 2.0:
                        label_text = "Trafik Levhasi"
                        bbox.color = [0.8, 0.8, 0.2]
                    else:
                        label_text = "Engel"
                        bbox.color = [0.5, 0.5, 0.5]

            # Basit Hız Tahmini (Önceki kare ile kıyaslama)
            obj_id = f"id_{i}"
            speed = 0.0
            if obj_id in prev_objs:
                old_pos, _ = prev_objs[obj_id]
                move_dist = np.linalg.norm(center - old_pos)
                speed = (move_dist / 0.1) * 3.6  # KITTI 10 FPS (0.1sn) kabulu

            prev_objs[obj_id] = (center, bin_path)

            # Metadata string formatı (Arayüzde parçalanarak kullanılır)
            full_meta = f"{label_text}|{speed:.1f}"
            bboxes.append(bbox)
            metadata.append(full_meta)

    return objects, bboxes, metadata


def detect_objects_2d(frame):
    """Kamera görüntüsü üzerinde YOLOv11 ile nesne tespiti yapar."""
    results = model_2d(frame, verbose=False)
    return results[0].plot()