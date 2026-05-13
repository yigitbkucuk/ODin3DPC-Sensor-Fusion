import open3d as o3d
import numpy as np


def create_demo_pc():
    # Rastgele bir nokta bulutu oluşturuyoruz (Gerçek veri gelene kadar yer tutucu)
    # 1000 tane nokta, 3 koordinat (x, y, z)
    points = np.random.rand(1000, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Noktaları renklendirelim (DL hocası görseli sever)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Tatlı bir mavi

    return pcd


def show_point_cloud(pcd):
    print("Görselleştirme penceresi açılıyor... Fare ile döndürebilirsin.")
    o3d.visualization.draw_geometries([pcd],
                                      window_name="ODin3DPC - İlk Adım",
                                      width=800, height=600)


if __name__ == "__main__":
    demo_cloud = create_demo_pc()
    show_point_cloud(demo_cloud)