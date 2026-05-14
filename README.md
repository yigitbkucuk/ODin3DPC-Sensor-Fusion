# ODin3DPC: Autonomous Vision System

Bu proje, otonom araç sistemlerinde çevresel farkındalığı artırmak amacıyla geliştirilmiş kapsamlı bir sensör füzyon (Sensor Fusion) projesidir. Geliştirme süreci boyunca KITTI veri setine ait toplam **7824 adet** eşleştirilmiş veri (kamera, LiDAR ve kalibrasyon dosyaları) kullanılarak sistemin kararlılığı test edilmiştir. 

Sistem mimarisinde, 2B kamera görüntüleri projeye özel olarak eğitilmiş (fine-tuned) **YOLOv11** ile nesne tespitinden geçirilirken; 3B LiDAR nokta bulutları (Point Cloud) **DBSCAN** kümeleme ve **RANSAC** düzlem segmentasyonu algoritmalarıyla uzamsal olarak analiz edilmektedir. Projeyi ziyaret eden kullanıcıların kodları anında ve sorunsuz test edebilmesi amacıyla sistem içerisine **`ornek_data`** klasörü entegre edilmiştir.

## Özellikler
* **Gerçek Zamanlı Füzyon:** 2D nesne tespiti ve 3D uzamsal farkındalığın tek bir karar mekanizmasında birleşimi.
* **Özelleştirilmiş Yapay Zeka:** Sınırlı donanım ile KITTI veri setine özel olarak "Fine-Tuning" edilmiş YOLO modeli.
* **Coğrafi Isı Haritası & Segmentasyon:** Gelişmiş 3D görselleştirme modları.
* **Akıllı Etkileşim:** PyQt6 ve Open3D tabanlı, çok iş parçacıklı (multi-threaded) kullanıcı arayüzü.

## Örnek Veri Seti (Demo) 
Devasa veri setlerini indirmekle uğraşmadan projeyi anında test edebilmeniz için ana dizine **`ornek_data`** adında özel bir klasör bırakılmıştır. Bu klasör içerisinde:
* `images_` (2B Kamera Görüntüleri)
* `velodyne_` (3B LiDAR Nokta Bulutları)
* `calib_` (Sensör Kalibrasyon Matrisleri)

olmak üzere 10'ar adet eşleştirilmiş örnek KITTI verisi bulunmaktadır. Arayüzü başlattıktan sonra "Veri Seti Yükle" butonuna basıp bu `ornek_data/velodyne_` klasöründeki dosyalardan herhangi birini seçerek sensör füzyon sisteminin nasıl çalıştığını canlı olarak inceleyebilirsiniz.
## Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu bilgisayarınıza indirin:
```bash
git clone [https://github.com/yigitbkucuk/ODin3DPC.git](https://github.com/yigitbkucuk/ODin3DPC.git)
cd ODin3DPC
```

2. Gerekli kütüphaneleri yükleyin:
```
pip install -r requirements.txt
```

3. Uygulamayı başlatın:
```
python main.py
```

## Teknik Rapor
Projenin donanım kısıtlamaları, algoritma mimarisi, yaşanan zorluklar (C++ kütüphane çakışmaları) ve performans metriklerini detaylandıran IEEE formatındaki teknik analize ana dizinde bulunan YigitBugraKucuk_ODin3DPC_Analiz_Raporu.pdf dosyasından ulaşabilirsiniz.

## 🎥 Proje Tanıtım Videosu

Arayüzün, 2D YOLO nesne tespitinin ve 3D LiDAR nokta bulutu segmentasyonunun eşzamanlı nasıl çalıştığını görmek için aşağıdaki videoya tıklayabilirsiniz:

[![ODin3DPC Demo](https://img.youtube.com/vi/https://www.youtube.com/watch?v=xMrrAmF25gI/maxresdefault.jpg)](https://www.youtube.com/watch?v=https://www.youtube.com/watch?v=xMrrAmF25gI)

## Hazırlayan
* Yiğit Buğra Küçük
* 230212048
* Yapay Zeka Mühendisliği
