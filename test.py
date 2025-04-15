from ultralytics import YOLO

# Load model from YAML file
model = YOLO(r'D:\test_ultra\ultralytics\yaml_file\yolov12_mamba.yaml')


# Huấn luyện mô hình với cấu hình dữ liệu
model.train(data=r'E:\ca\gan nhan.v7-loet.yolov8\data.yaml',
            epochs=1,  # Số lượng epoch
            imgsz=64, # Kích thước ảnh đầu vào
            save_period=5,
            device='cpu')  # train trên 2 cpu 0 và 1 (train trên 1 cpu device=0)