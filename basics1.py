from ultralytics import YOLO
import cv2
import cvzone
import torch
import numpy as np
import math
#model = torch.hub.load("ultralytics/yolov5","yolov5n")
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(3,800)
cap.set(4,500)

classname = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
while True:
    ret,frame = cap.read()
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(frame,(x1,y1,w,h))
            confidence = math.ceil(box.conf[0]*100)/100
            index = int(box.cls[0])
            cvzone.putTextRect(frame,f"{classname[index]} {confidence}",(x1,y1-20),scale=1,thickness=1)

    cv2.imshow("Frame",frame)
    cv2.waitKey(1)