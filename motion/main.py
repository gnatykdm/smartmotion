import cv2
import time
import keyboard
import pyautogui
import numpy as np

from handtracker import HandTrackingDynamic

MIN_DISTANCE = 20
SHOW_DRAW = False

FRAME_WIDTH = 480
FRAME_HEIGHT = 240

GESTURE_MODE = True
SWITCH_DURATION = 2

def main():
    global GESTURE_MODE
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic(maxHands=1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    screen_width, screen_height = pyautogui.size()

    click_down = False
    double_click_delay = 0.5
    last_double_click_time = 0

    prev_x, prev_y = 0, 0
    smoothening = 7

    gesture_switched = False
    gesture_hold_start_time = None

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = detector.findFingers(frame)
        lmsList, _ = detector.findPosition(frame, draw=SHOW_DRAW)

        fingers = []

        if len(lmsList) != 0:
            fingers = detector.fingersUp()
            print(f"FINGERS: {fingers}")
        if fingers == [0, 1, 1, 0, 0]:
            if gesture_hold_start_time is None:
                gesture_hold_start_time = time.time()

            held_duration = time.time() - gesture_hold_start_time

            if held_duration >= SWITCH_DURATION and not gesture_switched:
                GESTURE_MODE = not GESTURE_MODE
                gesture_switched = True
                print(f"[INFO] - GESTURE_MODE {'ENABLED' if GESTURE_MODE else 'DISABLED'}")
        else:
            gesture_hold_start_time = None
            gesture_switched = False

        if GESTURE_MODE and len(lmsList) != 0:
            x_index, y_index = lmsList[8][1], lmsList[8][2]
            target_x = int(np.interp(x_index, [0, FRAME_WIDTH], [0, screen_width]))
            target_y = int(np.interp(y_index, [0, FRAME_HEIGHT], [0, screen_height]))
            curr_x = prev_x + (target_x - prev_x) // smoothening
            curr_y = prev_y + (target_y - prev_y) // smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            distance, _, _ = detector.findDistance(4, 8, frame, draw=SHOW_DRAW)
            if distance < MIN_DISTANCE:
                if not click_down:
                    click_down = True
                    pyautogui.mouseDown()
                    print("[INFO] - Left Clicked!")
            else:
                if click_down:
                    click_down = False
                    pyautogui.mouseUp()
                    print("[INFO] - Mouse Released!")

            # Double click (thumb + middle)
            double_distance, _, _ = detector.findDistance(4, 12, frame, draw=SHOW_DRAW)
            if double_distance < MIN_DISTANCE:
                if time.time() - last_double_click_time > double_click_delay:
                    pyautogui.doubleClick()
                    print("[INFO] - Double Clicked!")
                    last_double_click_time = time.time()

            # Right click (thumb + ring)
            right_distance, _, _ = detector.findDistance(4, 16, frame, draw=SHOW_DRAW)
            if right_distance < MIN_DISTANCE:
                pyautogui.rightClick()
                print("[INFO] - Right Clicked!")

            # Music control (thumb + pinky)
            music_distance, _, _ = detector.findDistance(4, 20, frame, draw=SHOW_DRAW)
            if music_distance < MIN_DISTANCE:
                keyboard.send("play/pause media")
                print("[INFO] - Music Play/Pause!")

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        mode_text = f"MODE: {'GESTURE' if GESTURE_MODE else 'NORMAL'}"
        cv2.putText(frame, mode_text, (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('Hand Mouse Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("[INFO] - Starting Hand Mouse Control...")
    main()
