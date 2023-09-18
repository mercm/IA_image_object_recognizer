import numpy as np
import argparse
import time
import cv2
import os
import sys
from Check_recono_obj import Check as ck

def import_image(image_path):
    image = cv2.imread(image_path)
    orig_file_name = os.path.basename(image_path)
    filename, ext = orig_file_name.split(".")

    #create blob. Normalize, scale and reshape image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (WIDTH, HEIGHT), swapRB=True, crop=False)

    ck.check_image(image, blob)

    return blob, filename, ext, image

def print_boxes(idx, boxes, colors, class_ids, labels, confidences, filename, ext, image):
    for i in idx:
        #get box coordinates
        x = boxes[i][0]
        y = boxes[i][1]
        w = boxes[i][2]
        h = boxes[i][3]
        #print box on image
        label = labels[class_ids[i]]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=THICKNESS)
        text = label + ": " + format(confidences[i],".2f")
        #print box behind label to make it easier to read
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, thickness=THICKNESS)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        cv2.rectangle(image, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        #change opacity of label's box
        image = cv2.addWeighted(image, 0.6, image, 0.4, 0)
        #print label on image
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, color=(0, 0, 0), thickness=THICKNESS)
        
    #save image
    cv2.imwrite(SAVE_PATH + filename + "_detections." + ext, image)
        
    return image

def main():
    #derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])

    #load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])
    labels = open(labelsPath).read().strip().split("\n")

    #initialize a list of colors to represent each possible class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    #load YOLO object detector model
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    #create blob
    blob, filename, ext, image_or = import_image(IN_IMAGE)
    h, w = image_or.shape[:2]

    #sets the blob as the input of the network
    net.setInput(blob)

    #get all the layer names, feed forward (inference) and get the network output
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    layers_out = net.forward(ln)

    #iterate over the neural network outputs and discard any object with confidence less tan the confidence parameter
    boxes = []
    confidences = []
    class_ids = []
    #loop over layer outputs
    for layer in layers_out:
        #loop over object detections
        for detection in layer:
            ##ck.check_detection(detection)

            #get confidence and label (class id) from detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #keep only those over de confidence threshold
            if confidence > CONFIDENCE:
                #resize boxes according to image size: detection[:4] == [center_x, center_y, width, height]
                box = detection[:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                #calculate top left corner of the box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                
                #save information
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    ##image = print_boxes(range(len(boxes)), boxes, colors, class_ids, labels, confidences, filename, ext, image_or)

    #non-maximum suppression and print detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    #check if there is any object detected
    if len(indexes) > 0:
        image = print_boxes(indexes.flatten(), boxes, colors, class_ids, labels, confidences,filename, ext, image_or)
    else:
        print("No objects detected in image " + filename + "." + ext)

    print("fin")



#path to input image
IN_IMAGE="./Unidad 4. Aplicaciones casos practicos/Practica reconocedor objetos/halloween.jpg"
#path to output image
#OUT_IMAGE="./Unidad 4. Aplicaciones casos practicos/Practica reconocedor objetos/shrekOut.jpg"
#path for saving created image
SAVE_PATH="./Unidad 4. Aplicaciones casos practicos/Practica reconocedor objetos/"
#width of network's input image
WIDTH=416
#height of network's input image
HEIGHT=416
#base path to YOLO directory
YOLO_PATH="./Unidad 4. Aplicaciones casos practicos/Practica reconocedor objetos/data/yolov3"
#minimum probability to filter weak detections
CONFIDENCE=0.5
#threshold used to filter boxes by score (if not reaching the minimum, the box is deleted)
SCORE_THRESHOLD=0.5
#threshold when applying non-maxima suppression (1 = boxes are identical. 0 = boxes not even intersected.)
IOU_THRESHOLD=0.3
#font size for detected object's labels
FONT_SCALE=0.5
#font thickness for detected object's labels
THICKNESS=1

main()

