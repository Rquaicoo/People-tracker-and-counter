

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import imutils, argparse
import numpy as np
from numpy.lib.utils import info
from imutils.video import VideoStream
from imutils.video import FPS
import time, dlib, cv2, datetime



input = str(input("Enter the video location or press 'w' to initialize the webcam"))

if input == "w":
    print("[INFO] Starting the live stream..")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    print("[INFO] Starting the video..")
    vs = cv2.VideoCapture(input)

h = 480
w = 640
frameArea = h*w
areaTH = frameArea/250

W, H = None, None

# instantiating a centroid tracker, then initialize a list and dictionary for storing each correlation
# and a dictionary to map each object ID to a unique object
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# intializing processed frames and number of objects that have moved up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator for benchmarking
fps = FPS().start()

# Keeps track of already written data (to prevent duplicate entries)
written_values = []

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX


while True:
    # grab the next frame and handle
    if input == "w":
        frame = vs.read()
    else:
        frame = vs.read()
        frame = frame[1]
    # video is ended if no frame is grabbed
    if input is not None and frame is None:
        break
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
        
        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print( 'UP:',totalUp)
        print ('DOWN:',totalDown)
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]    
    contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            status = "Waiting"
            rects = []

                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker

            new = True
            
            #15 is the number of skip frames
            if totalFrames % 15 == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []
                new = False
        
                startX, startY, endX, endY = cv2.boundingRect(cnt)
                box = np.array([startX,startY,endX,endY])
                (startX, startY, endX, endY) = box.astype('int')
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startY, startX, endY + startY, endX + startX)                
                
                tracker.start_track(rgb, rect)
                
                # add tracker to the list of trackers
                trackers.append(tracker)
                
            else:
                # looping over the trackers
                for tracker in trackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # updating the tracker and position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # adding the bounding box coordinates
                    rects.append((startY, startX, endY + startY, endX + startX))
            # drawing the horizontal line
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)

            # using the centroid tracker to update old centroid with the newly computed ones
            objects = ct.update(rects)

            # looping over tracked objects
            for (objectID, centroid) in objects.items():

                # check if a trackable object exists for the current one
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # create one if none exists
                else:
                    y = [c[1] for c in to.centroids]
                    # direction is determined by the difference between current and previous y coordinates
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # determining direction by movement towards the line and change in y-axis
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0:
                            totalUp += 1
                            to.counted = True

                        elif direction > 0:
                            totalDown += 1
                            to.counted = True

                # storing the trackable object
                trackableObjects[objectID] = to

                # writing the object ID
                text = "ID {}".format(objectID)
            
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame,(startY,startX),(endY + startY,endX + startX),(0,255,0),2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

            # tuple of info on the screen
            info = [
                ("Exiting", totalUp),
                ("Entering", totalDown),
                ("status", status),
            ]

            # Writing the output to a csv file
            if status.lower() != "waiting" and written_values != [totalUp, totalDown]:
                written_values = [totalUp, totalDown]
                timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                with open("./results.csv", "a") as file:
                    result = f"{timestamp},{totalUp},{totalDown}\n"
                    file.write(str(result))

                # drawing our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame,text,(10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2,)

                cv2.putText(
                frame,
                "-Prediction border--",
                (10, H - ((20) + 200)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            




    cv2.imshow("Realtime Object Detection and Tracking", frame)
    cv2.imshow('Mask',mask)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    totalFrames += 1

    # updating the frames per second info
    fps.update()
fps.stop()
print("Elapsed time {:.2f}".format(fps.elapsed()))
print("Approx. FPS {:.2f}".format(fps.fps()))
print(totalUp)
print(totalDown)

try:
    vs.release()
except:
    vs.stop()
cv2.destroyAllWindows()
