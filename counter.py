from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from numpy.lib.utils import info
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import timedelta
import imutils
import numpy as np
import argparse
import time
import dlib
import cv2
import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help ="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help ="path to Caffe 'pre-trained model")
ap.add_argument("-i", "--input", type =str,help ="path to optional input video file")
ap.add_argument("-o", "--output", type =str, help ="path to output video file. It is optional")
ap.add_argument("-c", "--confidence", type =float, default = 0.4, help ="probability threshold for detections")
ap.add_argument("-s", "--skip_frames", type =int, default=30, help ="number frames skippped between detections")

args = vars(ap.parse_args()) #parsing the arguments

#MOBILENET SSD
#initializing the classes for mobilenet
CLASSSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "mototbike", "person", "pottedplant", "sheep", "sofa",
"train", "tvmonitor"]

#initializing mobilenet
mobilenet = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#testing the model with or video or initializing the webcam if none is supplied
if not args.get("input", False):
    print("initializing video stream")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    print("opening video file...")
    vs = cv2.VideoCapture(args["input"])

#video writer
writer = None

#frame dimensions
W, H = None, None

#instatiating a centroid tracker, a list and dictionary for storing each correlation
#and a dictionary to map each object ID to a unique object
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

#intializing processed frames and number of objects that have moved up or down
totalFrames = 0
totalDown = 0
totalUp = 0

#frames per second throughput estimator for benchmarking
fps = FPS().start()

# Keeps track of already written data (to prevent dublicate entries)
written_values = []

while True:
    #grab the next frame and handle
    frame= vs.read()
    frame = frame[1] if args.get("input", False) else frame
    #video is ended if no frame is grabbed
    if args["input"] is not None and frame is None:
        break

    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #set the frame if they are empty
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    #writer is intitialized if video is written to a disk
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        writer = cv2.VideoWriter(args["output"], fourcc, 30 ,(W, H), True)


    status = "Waiting"
    rects = []

    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting"
        trackers = []

        #performing inferences
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        mobilenet.setInput(blob)
        detections = mobilenet.forward()

        #looping over the detections
        for i in np.arange(0, detections.shape[2]):
            #extracting the probability for each prediction
            confidence = detections[0, 0, i, 2]

            #filtering weak predictions
            if confidence > args["confidence"]:
                index = int(detections[0, 0, i, 1])

                #ignoring the class if it is not person
                if CLASSSES[index] != "person":
                    continue

                #computing the bounding box cooordinates
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                #constructing a rectangle from the biunding box
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                #add tracker to the list of trackers
                trackers.append(tracker)
    #utilizing trackers rather than detectors for higher frame processing throughput
    else:
        #looping over the trackers
        for tracker in trackers:
            status = "Tracking"

            #updating the tracker and position
            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            #adding the bounding box coordintes 
            rects.append((startX, startY, endX, endY))
    #draawing the horizontal line        
    cv2.line(frame, (0, H // 2), (W, H //2), (0, 0, 255), 4)

    #using the centroid tracker to update old centroid swith the newly computed ones
    objects = ct.update(rects)

    #looping over tracked objects
    for (objectID, centroid) in objects.items():

        #check if a trackable object exists for the current one
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        
        #create one if none exists
        else:
            y = [c[1] for c in to.centroids]
            #direction is determined by the difference between current and previous y coordinates
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            #determining direction by movement towards the line and change in y-axis 
            if not to.counted:
                if direction < 0 and centroid[1] < H // 2:
                    totalUp +=1
                    to.counted = True

                elif direction > 0 and centroid[1] > H // 2:
                    totalDown +=1
                    to.counted = True
        
        #storing the trackable object
        trackableObjects[objectID] = to

        #writing the object ID
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

    #tulpe of info on the screen
    info  = [
        ("up", totalUp),
        ("down", totalDown),
        ("status", status),
    ]

    # Writing the output to a csv file
    if status.lower() != 'waiting' and written_values != [totalUp, totalDown]:
        written_values = [totalUp, totalDown]
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open("./results.csv", "a") as file:
            result = f"{timestamp},{totalUp},{totalDown}\n"
            file.write(str(result))

    #drawing our frame
    for (i, (k,v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 255), 2)

    #writing the frame if necessary
    if writer is not None:
        writer.write(frame)

    #displaying the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    totalFrames +=1

    #updating the frames per second info
    fps.update()
fps.stop()
print("elapsed time {:.2f}".format(fps.elapsed()))
print("approx. FPS {:.2f}".format(fps.fps()))
print(totalUp)
print(totalDown)
if writer is not None:
    writer.release()

if not args.get("input", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()

#run this code in command line interface to view the model at work with a sample video
#use the path without the input to initialize the webcam
#python counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4
