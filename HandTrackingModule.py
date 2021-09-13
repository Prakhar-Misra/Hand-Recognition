import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands
import math


class HandTracker():
    ''' This module is a custom module that works upon Mediapipe. I can be used to create various programs and applications that require Hand Guesture Recognition '''
    def __init__(self, mode = False, maxHands = 2, detectionConf = 0.5, trackingConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackingCong = trackingConf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackingCong)
        self.mpDraw = mp.solutions.drawing_utils

    def handsMap(self, img, draw = True):
        '''It will detect the hand and map the points on the hand. The values it takes are:
            1. img: it is the image itself or the recording you are doin at the minute
            2. draw: this will show that mappings in the video itself. Set it to 'False' if you don't want to see the mapping image in realtime'''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print (results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:                   
                    self.mpDraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def findPossition(self, img, handNo = 0):
        ''' This will find a specific point of the hand that has been mapped. It takes inputs as:
            1. img: the image or real time video itself
            2. handNo: takes value of 0 and 1 which refers to right and left hand respectively'''
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
        ''' Finds Which finger is up and which finger is closed into the fist. It returns the value in a list of 5 boolean values(0 for down or 1 for up) and is as follows:
            1. The first value or the 0 index is the value of thumb
            2. second value is the value of index
            3. Third is the value of middle one
            4. fourth is the value of ring
            5. fifth is the value of little (pinky) finger'''
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
        ''' Finds the Distance between any 2 points f1 and f2 marked on the hand using the handMap function.
            the draw value (if true) will draw a line in between the 2 points and the center of that line.'''
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
