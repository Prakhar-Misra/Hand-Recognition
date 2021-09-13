import cv2
import mediapipe
import HandTrackingModule as htm
import autopy
import numpy as np


cam = cv2.VideoCapture(0)

cam.set(3,640)
cam.set(4,480)

detector = htm.HandTracker(maxHands=1)
frameR = 100
smooth_movement = 7
plocX, plocY = 0,0
clocX, clocY = 0,0

wScr, hScr = autopy.screen.size()
#print(wScr, hScr)

while True:
    s , img = cam.read()

    # find Landmarks
    img = detector.handsMap(img=img)
    lmlist = detector.findPossition(img)

    # get them fingertips

    if len(lmlist) > 0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

        # ckeck for fingers that are up
        fingers = detector.FingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (640-frameR, 480-frameR), (255,0,255), 2)

        # middle fingers moving mode
        if fingers[2] == 1:

            #covert cordinates
            x3 = np.interp(x2, (frameR,640-frameR), (0, wScr))
            y3 = np.interp(y2, (frameR,480-frameR), (0, hScr))

            #smoothen values
            clocX = plocX + (x3-plocX) // smooth_movement
            clocY = plocY + (y3-plocY) // smooth_movement
            # move mouse pointer
            autopy.mouse.move(wScr- clocX, clocY)
            plocX, plocY = clocX, clocY

        #index finger is select mode
        if fingers[1] == 1 :

            # distance between fingers
            # length, img, length_line_info = detector.FindDistance(8,7, img, draw=False)
            # print(length)
            if lmlist[8][2] > lmlist[7][2]:

                # click mouse of distance short
                autopy.mouse.click()

    #next step was FPS, I don't care. sorry!

    cv2.imshow("IMG", img)
    cv2.waitKey(1)