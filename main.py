from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')


video_path = './test3.mp4'
cap = cv2.VideoCapture(video_path)
ret = True

while ret:
    ret, frame = cap.read()

    resulat = model.track(frame, persist=True)

    frame_ = resulat[0].plot()
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) == ord("q"):
         break

