import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands
import math


class HandTracker():
    def __init__(self, mode = False, maxHands = 2, detectionConf = 0.5, trackingConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackingCong = trackingConf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackingCong)
        self.mpDraw = mp.solutions.drawing_utils

    def handsMap(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print (results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:                   
                    self.mpDraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def findPossition(self, img, handNo = 0, draw = True):
        self.lmlist =[]

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(cx,cy)
                self.lmlist.append([id, cx, cy])

        return self.lmlist

    
    def FingersUp(self):
        tipIds = [4,8,12,16,20]
        fingers = []
        if self.lmlist[tipIds[0]][1] > self.lmlist[tipIds[0]+1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1,5):
            if self.lmlist[tipIds[id]][2] < self.lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # totalFingers = sum(fingers)
        return fingers
    
    def FindDistance(self, f1, f2, img, draw=False):
        x1,y1 = self.lmlist[f1][1], self.lmlist[f1][2]
        x2,y2 = self.lmlist[f2][1], self.lmlist[f2][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.circle(img, (x1,y1), 15, (255,0,255),cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (255,0,255),cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (255, 0, 255), 3)
            cv2.circle(img, (cx,cy), 15,(255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1,x2,y1,y2,cx,cy]
def main():
    cap = cv2.VideoCapture(0)

    detector = HandTracker()

    while True:
        success, img = cap.read()
        img = detector.handsMap(img)
        lmlist = detector.findPossition(img)
        #if len(lmlist) != 0:    
            #print(lmlist[])
        if(len(lmlist)>0):
            fingers = detector.FingersUp()
            print(fingers)

        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main() 
