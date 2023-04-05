import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/5.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
while True:
    success, img = cap.read()

    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRBG)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #draw the bounding box
            mpDraw.draw_detection(img, detection)

            #print(id, detection)
            #print(detection.score)
            ##print(detection.location_data.relative_bounding_box)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Video", img)
    cv2.waitKey(1)


