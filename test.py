from ultralytics import YOLO

# Load model from YAML file
model = YOLO(r'D:\test_ultra\ultralytics\yaml_file\yolov12_mamba.yaml')

# Hiển thị thông tin mô hình
print(model)
# Hoặc sử dụng phương thức info để xem chi tiết cấu trúc mô hình
model.info()
