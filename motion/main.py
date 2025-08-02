import cv2
import mediapipe as mp
import time
import math
import pyautogui
import numpy as np

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lmsList, bbox

    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        x1, y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]

def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    screen_width, screen_height = pyautogui.size()
    click_down = False
    double_click_delay = 0.5
    last_double_click_time = 0

    prev_x, prev_y = 0, 0
    smoothening = 7

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = detector.findFingers(frame)
        lmsList, bbox = detector.findPosition(frame, draw=False)

        if len(lmsList) != 0:
            x_index, y_index = lmsList[8][1], lmsList[8][2]
            x_thumb, y_thumb = lmsList[4][1], lmsList[4][2]

            target_x = int(np.interp(x_index, [0, 640], [0, screen_width]))
            target_y = int(np.interp(y_index, [0, 480], [0, screen_height]))

            curr_x = prev_x + (target_x - prev_x) // smoothening
            curr_y = prev_y + (target_y - prev_y) // smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Left click (thumb + index)
            distance, _, _ = detector.findDistance(4, 8, frame, draw=True)
            if distance < 40:
                if not click_down:
                    click_down = True
                    pyautogui.mouseDown()
            else:
                if click_down:
                    click_down = False
                    pyautogui.mouseUp()

            # Double click (thumb + middle)
            double_distance, _, _ = detector.findDistance(4, 12, frame, draw=True)
            if double_distance < 40:
                if time.time() - last_double_click_time > double_click_delay:
                    pyautogui.doubleClick()
                    last_double_click_time = time.time()

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Hand Mouse Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()