import cv2
import HandTrackingModule as htm
import autopy
import numpy as np


cam = cv2.VideoCapture(0)

cam.set(3,640)
cam.set(4,480)

detector = htm.HandTracker(maxHands=1) # AN OBJECT OF HAND TRACKING MODULE
frameR = 100
smooth_movement = 7 # ALTER THIS TO MAKE THE MOVEMENT OF THE CURSOR FASTER(LOWER VALUE) OR SLOWER (HIGHER VALUE)
plocX, plocY = 0,0
clocX, clocY = 0,0

wScr, hScr = autopy.screen.size()
#print(wScr, hScr)

while True:
    s , img = cam.read()

    # FINDING HAND LANDMARKS (THE POINTS ON HAND)
    img = detector.handsMap(img=img)
    lmlist = detector.findPossition(img)


    if len(lmlist) > 0:
        # THESE TWO ARE THE X AND Y COODINATES TO THE TIPS OF INDEX(8) AND MIDDLE(12) FINGERS
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

        # CHECK FOR FINGERS THAT ARE UP
        fingers = detector.FingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (640-frameR, 480-frameR), (255,0,255), 2)

        # THE MOVEMENT OF THE MOUSE IS UPON THE MOVEMENT OF THE MIDDLE FINGER
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

        # WHEN YOU PUT THE INDEX FINGER DOWN IT WILL TRIGGER THE MOUSE CLICK EFFECT
        # FOR BETTER EXECUTION PUT BOTH FINGERS UP AND PUT THE INDEX FINGER DOWN ONLY WHEN YOU WANT TO CLICK
        if fingers[1] == 0 : 
            autopy.mouse.click()

    cv2.imshow("IMG", img)
    cv2.waitKey(1)
