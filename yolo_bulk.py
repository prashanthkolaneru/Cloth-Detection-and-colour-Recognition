# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import numpy as np
import os
import time
import keras
import tensorflow
from color_detection import color_detection

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())
path = r'./crop/'
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")
dir_path = os.path.dirname(os.path.realpath(args['image'])) 
print(dir_path)
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolo-obj_15000.weights"])
configPath = os.path.sep.join([args["yolo"], "yolo-obj.cfg"])

# load our YOLO object detector trained on Custom dataset (8 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
index = 0
for item in os.listdir(dir_path):
    try:
        # Use os.path.join to construct the full path of the item
        image = os.path.join(dir_path, item)
        print('image_path',image)
        image_path = image


        image = cv2.imread(image)
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        predictions_local=[]
        center_boxes=[]
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    center_boxes.append([centerX, centerY, width, height])
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        print("classIDs: ",classIDs)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])
        ids=[]
        for i in range(0,len(idxs)):
            print(idxs[i])

            ids.append(idxs[i])
        #print("IDS: ",ids)
        print("IDS:",ids)
        color_class=[]
        #for k1 in range(0,len(idxs)):
        #	color_class.append([])
        for k in range(0,len(ids)):
            f=ids[k]
            #print("{}:{}:{}:{}".format(idxs[k],LABELS[classIDs[f]],confidences[f],center_boxes[f]))
            centerX= center_boxes[f][0]
            centerY= center_boxes[f][1]
            width= center_boxes[f][2]
            height= center_boxes[f][3]
            color_image=image[centerY-int(height/2):centerY+int(height/2),centerX-int(width/2):centerX+int(width/2)]

            predictions_local.append([LABELS[classIDs[f]], centerX, centerY, width, height])
        #print("idxs:",idxs)
        Color = (255, 0, 0)
        # ensure at least one detection exists
        
        if len(idxs) > 0:
            #print(len(idxs))
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                cv2.rectangle(image, (x, y), (x + w, y + h), (Color), 2)
                
                cropped = image[y+10:y + h-10, x+10:x + w-10]
                #cv2.imwrite(path  + str(index) + LABELS[classIDs[i]] + '.jpg',cropped)
                
            
                requested_colour1,closest_name = color_detection(cropped)
                print(" closest colour name:", closest_name)
                print(LABELS[classIDs[i]])
                print(confidences[i])
                
                text = "{}  {},{} ".format(LABELS[classIDs[i]],requested_colour1,closest_name)
                height, width, channels = image.shape
                #print(height,width)

                # Set the font scale based on the image size
                if height > 1000 or width > 1000:
                    font_scale = 1.5
                    thickness = 3
                else:
                    font_scale = 0.3
                    thickness = 1

                cv2.putText(image, text, (x-30, y - 5), cv2.FONT_HERSHEY_SIMPLEX,font_scale, Color, thickness)

        image_id = image_path.split('\\')[-1].split('.')[0]
        image_type = image_path.split('\\')[-1].split('.')[1]
        output_path = '.' + '/' + args['output'] + '/' + image_id + '.' + image_type 
        print(output_path)
        cv2.imwrite(output_path,image)
        index = index + 1
    except cv2.error as e:
        print('Cloth is not detected')