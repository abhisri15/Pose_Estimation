import cv2
import mediapipe as mp
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseEstimationVideos/1.mp4')
pTime = 0
cTime = 0

detector = pm.poseDetector()

while True:
    success, img0 = cap.read()
    img = pm.rescaleFrame(img0, scale=0.3)
    img = detector.findPose(img)

    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0,0,255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)  # delay = 1millisecond
