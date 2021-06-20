import cv2
import numpy as np
import handTrackingModule as Htm
import time
import autopy

###############################
wCam, hCam = 640, 480
frameRed = 100  # frame Reduction
Smoothing = 10
###############################

P_time = 0
P_Loc_X, P_Loc_Y = 0, 0
C_Loc_X, C_Loc_Y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = Htm.HandDetector(maxHands=1)
W_Scr, H_Scr = autopy.screen.size()
# 1536.0 864.0
while True:
    # 1. Find Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # print(lmList)
    # 2. Get the tips of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameRed, frameRed), (wCam - frameRed, hCam - frameRed), (255, 0, 255), 2)

        # 4. Only Index finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates

            x3 = np.interp(x1, (frameRed, wCam-frameRed), (0, W_Scr))
            y3 = np.interp(y1, (frameRed, hCam-frameRed), (0, H_Scr))

            # 6. Smoothen Values
            C_Loc_X = P_Loc_X + (x3 - P_Loc_X) / Smoothing
            C_Loc_Y = P_Loc_Y + (y3 - P_Loc_Y) / Smoothing

            # 7. Move Mouse
            autopy.mouse.move(C_Loc_X, C_Loc_Y)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            P_Loc_X, P_Loc_Y = C_Loc_X, C_Loc_Y

        # 8. When Both index amd middle finger are up then it is in clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find the distance between fingers
            length, img, lineInfo = detector.findDistance(img, 8, 12)
            print(length)
            if int(length) < 45:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                # 10. click move if distance short
                autopy.mouse.click()

    # 11. Frame Rate
    C_time = time.time()
    fps = 1/(C_time-P_time)
    P_time = C_time
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
