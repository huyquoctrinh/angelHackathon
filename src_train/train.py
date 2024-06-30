from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
model.train(
    data="/home/nhdang/test/test/data.yaml",
    # warmup_epochs = 3,
    batch = 16,
    imgsz = 640,
    epochs=200,
    optimizer = "AdamW",
    workers = 8,
    lr0 = 1e-4,
    mosaic = 0.5,
    mixup = 0.5,
    copy_paste = 0.5,
    flipud = 0.2,
    device = "1",
    close_mosaic = 10,
    project = "yolov9_wli"
)
