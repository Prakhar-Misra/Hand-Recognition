import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands


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
        lmlist =[]

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(cx,cy)
                lmlist.append([id, cx, cy])
        
        return lmlist
def main():
    cap = cv2.VideoCapture(0)

    detector = HandTracker()

    while True:
        success, img = cap.read()
        img = detector.handsMap(img)
        lmlist = detector.findPossition(img)
        #if len(lmlist) != 0:    
            #print(lmlist[])

        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()