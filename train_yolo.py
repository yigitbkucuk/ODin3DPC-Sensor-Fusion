from ultralytics import YOLO

def start_training():
    # Temel modeli yükle
    model = YOLO("yolo11n.pt")

    # Fine-tuning işlemini başlat
    model.train(
        data="data.yaml",
        epochs=30,         # 30 fotoğraf için 30 epoch idealdir
        imgsz=640,
        batch=4,
        name="odin_final_model"
    )

if __name__ == "__main__":
    start_training()