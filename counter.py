import numpy as np
import cv2 as cv
import Person
import time

try:
    log = open("log.txt", "w")
except:
    print("Cannot open log file")

# Entry and exit counters
cnt_up = 0
cnt_down = 0

# Video source
# # cap = cv.VideoCapture(0) # For webcam
input = str(input("Enter the video path:  "))

cap = cv.VideoCapture(input)


h = 480
w = 640
frameArea = h * w
areaTH = frameArea / 250
# print("Area Threshold", areaTH)

# Entry / exit lines
line_up = int(2 * (h / 5))
line_down = int(3 * (h / 5))

up_limit = int(1 * (h / 5))
down_limit = int(4 * (h / 5))

print("Red line y:", str(line_down))
print("Blue line y:", str(line_up))
line_down_color = (255, 0, 0)
line_up_color = (0, 0, 255)
pt1 = [0, line_down]
pt2 = [w, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))
pt3 = [0, line_up]
pt4 = [w, line_up]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [0, up_limit]
pt6 = [w, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
pt7 = [0, down_limit]
pt8 = [w, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

# Background subtractor
fgbg = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=70)

# Structuring elements for morphogenic filters
kernelOp = np.ones((3, 3), np.uint8)
kernelOp2 = np.ones((5, 5), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

# Variables
font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while cap.isOpened():
    # Read an image from the video source
    ret, frame = cap.read()

    for i in persons:
        i.age_one()  # age every person one frame

    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    # Binary to remove shadows (gray color)
    try:
        ret, imBin = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
        ret, imBin2 = cv.threshold(fgmask2, 200, 255, cv.THRESH_BINARY)
        # Opening (erode-> dilate) to remove noise.
        mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
        mask2 = cv.morphologyEx(imBin2, cv.MORPH_OPEN, kernelOp)
        # Closing (dilate -> erode) 
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernelCl)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernelCl)
    except:
        print("EOF")
        print("Exiting:", cnt_up)
        print("Entering:", cnt_down)
        break

    # CONTOURS

    contours0, hierarchy = cv.findContours(
        mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > areaTH:

            # TRACKING

            # Need to add conditions for multi-person, screen outputs and inputs.

            M = cv.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv.boundingRect(cnt)

            new = True
            if cy in range(up_limit, down_limit):
                for i in persons:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        # the object is close to one that has already been detected before
                        new = False
                        i.updateCoords(
                            cx, cy
                        )  # updates coordinates in object and resets age
                        if i.going_UP(line_down, line_up) == True:
                            cnt_up += 1
                            print(
                                "ID:",
                                i.getId(),
                                " exited",
                                time.strftime("%c"),
                            )
                            log.write(
                                "ID: "
                                + str(i.getId())
                                + " exited "
                                + time.strftime("%c")
                                + "\n"
                            )
                        elif i.going_DOWN(line_down, line_up) == True:
                            cnt_down += 1
                            print(
                                "ID:",
                                i.getId(),
                                "entered",
                                time.strftime("%c"),
                            )
                            log.write(
                                "ID: "
                                + str(i.getId())
                                + " entered "
                                + time.strftime("%c")
                                + "\n"
                            )
                        break
                    if i.getState() == "1":
                        if i.getDir() == "down" and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == "up" and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        # remove i from persons list
                        index = persons.index(i)
                        persons.pop(index)
                        del i  # free up i memory
                if new == True:
                    p = Person.MyPerson(pid, cx, cy, max_p_age)
                    persons.append(p)
                    pid += 1

            # drawings

            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv.drawContours(frame, cnt, -1, (0,255,0), 3) #optional, to view contour

    for i in persons:
        cv.putText(
            frame,
            str(i.getId()),
            (i.getX(), i.getY()),
            font,
            0.3,
            i.getRGB(),
            1,
            cv.LINE_AA,
        )

    # Information on count
    str_up = "Exiting: " + str(cnt_up)
    str_down = "Entering: " + str(cnt_down)
    frame = cv.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
    frame = cv.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
    frame = cv.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
    cv.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv.LINE_AA)

    cv.imshow("Frame", frame)
    # cv.imshow("Mask", mask) Optional to show mask

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
# END while(cap.isOpened())

# Clearing internal buffer of file
log.flush()
log.close()
cap.release()
cv.destroyAllWindows()
