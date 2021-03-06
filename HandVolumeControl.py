import cv2 
import numpy as np
import HandTrackingModule as htm
import math
from subprocess import call
import numpy as np

#######################
wCam, hCam = 640,480
#######################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

tracker = htm.HandTracker(detectionConf=0.5)

while True:
    success, img = cap.read()
    img = tracker.handsMap(img)
    lmlist = tracker.findPossition(img)
    if len(lmlist) > 0:
        #print(lmlist)

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 15, (255,0,255),cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255, 0, 255), 3)
        cv2.circle(img, (cx,cy), 15,(255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)

        vol = np.interp(length, [50,300], [0,100])
        #print(round(vol))
        call(["amixer", "-D", "pulse", "sset", "Master", str(int(vol))+"%"])

        if length < 30:
            cv2.circle(img, (cx,cy), 15,(0,255,0),3)


    cv2.imshow("IMG",img)
    cv2.waitKey(1)
