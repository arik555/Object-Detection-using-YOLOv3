import numpy as np
import argparse
import cv2
import os
import imutils
import imageio

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video")
args = vars(ap.parse_args())

labelsPath = os.path.join("file", "coco.names")
CLASSES = open(labelsPath).read().strip().split('\n')
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

weightsPath = "file/yolov3.weights"
configPath = "file/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(args["video"])
writer = None
(W, H) = (None, None)
 
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(cap.get(prop))
    print("[INFO] {} total frames in video".format(total))
 
# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

while True:
    _ret, frame = cap.read()

    if not _ret:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.6:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (width / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = COLORS[classIDs[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            disply_txt = '{:0.2f}% {}'.format(confidences[i], CLASSES[classIDs[i]])
            cv2.putText(frame, disply_txt, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if writer is None:
        codec = cv2.VideoWriter_fourcc(*"DIVX")
        writer = cv2.VideoWriter("outdir/output.mp4", codec, 30, (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("Single frame took {:0.3f} sec".format(elap))
            print("Completion Time {:0.3f} sec".format(elap*total))
    writer.write(frame)

writer.release()
cap.release()










