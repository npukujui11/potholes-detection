import multiprocessing
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# 解决OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.的问题

def main():
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='D:\\program\\potholes-detection\\dataset\\dataAA.v1i.yolov8\\data.yaml', epochs=100, imgsz=640)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
