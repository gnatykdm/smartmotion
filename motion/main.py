import cv2
import time
import keyboard
import pyautogui
import numpy as np

from handtracker import HandTrackingDynamic

MIN_DISTANCE = 30
SHOW_DRAW = False

FRAME_WIDTH = 480
FRAME_HEIGHT = 240

GESTURE_MODE = True

def main():
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
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
        lmsList, bbox = detector.findPosition(frame, draw=SHOW_DRAW)

        if len(lmsList) != 0:
            x_index, y_index = lmsList[8][1], lmsList[8][2]

            target_x = int(np.interp(x_index, [0, FRAME_WIDTH], [0, screen_width]))
            target_y = int(np.interp(y_index, [0, FRAME_HEIGHT], [0, screen_height]))

            curr_x = prev_x + (target_x - prev_x) // smoothening
            curr_y = prev_y + (target_y - prev_y) // smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Left click (thumb + index)
            distance, _, _ = detector.findDistance(4, 8, frame, draw=SHOW_DRAW)
            if distance < MIN_DISTANCE:
                if not click_down:
                    click_down = True
                    print("[INFO] - Left Clicked!")
                    pyautogui.mouseDown()
            else:
                if click_down:
                    click_down = False
                    print("[INFO] - Left Clicked!")
                    pyautogui.mouseUp()

            # Double click (thumb + middle)
            double_distance, _, _ = detector.findDistance(4, 12, frame, draw=SHOW_DRAW)
            if double_distance < MIN_DISTANCE:
                if time.time() - last_double_click_time > double_click_delay:
                    pyautogui.doubleClick()
                    print("[INFO] - Double Clicked!")
                    last_double_click_time = time.time()

            # Right Click
            right_distance, _, _ = detector.findDistance(4, 16, frame, draw=SHOW_DRAW)
            if right_distance < MIN_DISTANCE:
                print("[INFO] - Right Clicked!")
                pyautogui.rightClick()

            # Music Stop/Play
            music_distance, _, _ = detector.findDistance(4, 20, frame, draw=SHOW_DRAW)
            if music_distance < MIN_DISTANCE:
                print("[INFO] - Music Play/Pause!")
                keyboard.send("play/pause media")

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Hand Mouse Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    print("[INFO] - Starting Hand Mouse Control...")
    main()