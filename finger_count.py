import cv2
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

finger_count = ['0','1','2','3','4','5']

detector = htm.HandTracker()

tipIds = [4,8,12,16,20]
totalFingers = 0

while True:
    success, img = cap.read()
    img = detector.handsMap(img)
    lmlist = detector.findPossition(img, draw=False)

    if len(lmlist) != 0:
        fingers = []
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0]+1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = sum(fingers)
        
    cv2.putText(img,str(totalFingers),(50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,255), 3 )

    cv2.imshow('Image', img)
    cv2.waitKey(1)
