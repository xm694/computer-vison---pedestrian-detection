#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 07:22:26 2023

@author: xiaomingmo
"""

import cv2 as cv
import numpy as np
import sys

#define frame resize function
def resize(img1, padColor = 255):
    #get the dimensions of input image
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    ratio1 = round(w1/h1,2)
    
    #define the VGA size of rescaled image
    sh = 480
    sw = 600
    ratio2 = round(sw/sh,2)
    
    #decide interpolation method
    if h1>sh or w1>sw: #shrinking image
        interp = cv.INTER_AREA
    else: #stretching image
        interp = cv.INTER_CUBIC
    
    #calculate the new image depend on the ratio
    if ratio1 == ratio2:
        img_c = cv.resize(img1, (sw, sh))
    
    else:
        if ratio2 > ratio1: #new horizontal image
            new_h = sh
            new_w = np.round(new_h*ratio1).astype(int)
            pad_horz = float(sw - new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot= 0, 0
            
        else: #new vertical image
            new_w = sw
            new_h = np.round(float(new_w)/ratio1).astype(int)
            pad_vert = float(sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
            
        #set pad color
        if len(img1.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
            padColor = [padColor]*3
        
        #scale and pad image
        img_c = cv.resize(img1, (new_w, new_h), interpolation=interp)
        img_c = cv.copyMakeBorder(img_c, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=padColor)
    
    return img_c


#use median method to estimate background
def estimateBg(file):
    cap = cv.VideoCapture(file)
    #randomly select 30 frames
    frameIds = cap.get(cv.CAP_PROP_FRAME_COUNT)*np.random.uniform(size=30)
    
    #store selected frames in an array
    selected_frames=[]
    for fid in frameIds:
        cap.set(cv.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        selected_frames.append(frame)
        
    #calculate the median along the time axis
    medianFrame = np.median(selected_frames, axis=0).astype(dtype=np.uint8)
    estemated_bg = resize(medianFrame)
    
    return estemated_bg


def task1(videoFile):
    vid = cv.VideoCapture(videoFile)
    #create the 'VideoWriter()' object 
    out = cv.VideoWriter('task1_result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (480*2,600*2))
    
    #inintialize object count
    count = -1
    
    #create background subtractor objects
    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    #get the estemated background
    estemated_bg = estimateBg(videoFile)
    
    while vid.isOpened():
        count = count+1
        ppl = 0
        car = 0
        other = 0
        
        ret, frame = vid.read()
        
        if not ret:
            break
        
        #read frame and resize
        frame = resize(frame)
        #covert frame to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #apply gaussian blur
        blurred = cv.GaussianBlur(gray_frame, (7,7), 0) 
        #apply threshold
        #threshold = cv.threshold(blurred, 50, 155, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        #apply background subtractor on each frame
        fg_mask = backSub.apply(blurred)
        #using morphological operators to remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        #mono_col = cv.dilate(mask, kernel, iterations=3)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
        
        #putting mask upon frames
        #mask2 = cv.bitwise_not(mono_col)
        binary_mask = np.where(fg_mask>0, 255, 0).astype(np.uint8)
        mask2 = cv.bitwise_and(frame, frame, mask=binary_mask)
        #fin = frame - mask2
        final_obj = mask2.copy()

        #apply the component analysis function
        (totalLabels, labelsIds, values, centroid) = cv.connectedComponentsWithStats(binary_mask, 4, ltype=None)
        
        #calculate the objects
        for i in range(1, totalLabels):
            w = int(values[i, cv.CC_STAT_WIDTH])
            h = int(values[i, cv.CC_STAT_HEIGHT])
            ratio = w/h
            area = int(values[i, cv.CC_STAT_AREA])
            if area>30 and ratio<0.5:
                ppl = ppl+1
            elif area>80 and ratio>0.5:
                car = car+1
            else:
                other = other +1
        
        binary_obj = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)
        hor1 = cv.hconcat([frame, estemated_bg])
        hor2 = cv.hconcat([binary_obj,final_obj])
        output_video = cv.vconcat([hor1, hor2])
        
        #print the command line output
        if int(totalLabels) == 1:
            print ("Frame " + ("%04d" % count) + " : " + str(int(totalLabels)-1) + " objects")
        else:
            print ("Frame " + ("%04d" % count) + " : " + str(int(totalLabels)-1) + " objects ( " + str(ppl) +" persons, " +str(car)+" cars and "+ str(other) +" others )" )
        
        out.write(output_video)
        cv.imshow('Task1', output_video)
        
        #error handler
        if cv.waitKey(1) == ord('q'):
            break   
    
    vid.release()
    out.release()
    cv.destroyAllWindows()

#task1('Trafficlights_compressed.avi')

def task2(videoFile):
    #load the coco class
    with open('object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')
    
    #create a different color array 
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    
    #load the DNN model
    model = cv.dnn.readNet(model='frozen_inference_graph.pb', 
                           config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                           framework='TensorFlow')
    
    #capture the video
    vid = cv.VideoCapture(videoFile)
    #create the 'VideoWriter()' object
    #out = cv.VideoWriter('video_result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (480, 600))
    
    #detect objects in each fram of the video
    while vid.isOpened():
        #count = 0
        pedestrians = []
        #pedestrian_size = []
        class_ids = []
        confidences = []
        boxes = []
        
        ret, frame = vid.read() 
        if ret:
            frame = resize(frame)
            #create sub-frames for the output collage
            frame1 = frame.copy()
            frame2 = frame.copy()
            frame3 = frame.copy()
            frame4 = frame.copy()
            
            #create blob from frame
            blob = cv.dnn.blobFromImage(image=frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
            model.setInput(blob)
            output = model.forward()
            
            #loop over each of the detections
            for detection in output[0,0,:,:]:
                #extract the confidence of the detection
                confidence = detection[2]
                
                if confidence > .4:
                    #get class id and map to the class name
                    class_id = detection[1]
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    
                    scores = detection[5:]
                    _, _, _, max_id = cv.minMaxLoc(scores)
                    class_id = max_id[1]
                                      
                    #continue draw overlapping bounding box if class is person
                    if class_name == 'person' and scores[class_id]>.25:
                        #get the bounding box coordinates
                        box_x = detection[3]*600
                        box_y = detection[4]*480
                        #get the bounding box width and height
                        box_w = detection[5]*600
                        box_h = detection[6]*480
                        
                        #storing boxes
                        box = np.array([box_x, box_y, box_w, box_h])
                        boxes.append(box)
                        confidence = scores[class_id]
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        
                        
                        #frame2 video frame with overlapped bounding-boxes of detected pedestrians
                        #draw a rectangle around each detected object
                        cv.rectangle(frame2, (int(box_x), int(box_y)), (int(box_w), int(box_h)), color, thickness=2)
                        
                        #frame3 video frame with detected and tracked bounding-boxes
                        #using NMSboxes to eliminate multiple bounding boxes
                        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
                        
                        result_class_ids = []
                        result_confidences = []
                        result_boxes = []
                        selected_box_size = []
                        
                        for i in indexes:
                            result_confidences.append(confidences[i])
                            result_class_ids.append(class_ids[i])
                            result_boxes.append(boxes[i])
                            
                            
                            for j in range(len(result_class_ids)):
                                box = result_boxes[j]
                                cv.rectangle(frame3, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=2)
                                # put the class name text on the detected object
                                cv.putText(frame3, 'pedestrian', (int(box[0]), int(box[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1)
                                
                                #frame4 video frame with 3 detected and tracked pedestrians that are most closed to the camera
                                #according to the size of the bounding box to decide the most close to the camera
                                #the closer the bigger of the size
                                #get the size of the bounding boxes
                                w = box[2]
                                h = box[3]
                                box_size = int(w)*int(h)
                                selected_box_size.append(box_size)
                                
                                #get the dimensions of boxes
                                cx = int(int(box[0])+int(box[2])/2)
                                cy = int(int(box[1])+int(box[3])/2)
                                #storing detected pedestrians
                                pedestrians.append((w, h, cx, cy))
                                
                            #sort the tracked objects by size
                            sorted_pedestrians = sorted(selected_box_size, key=lambda x:box_size, reverse=True)
                            
                            for h in range(len(result_class_ids)):
                                box2 = result_boxes[h]
                                h_box_size = int(box2[2])*int(box2[3])
    
                                if h_box_size>=sorted_pedestrians[0] or h_box_size>=sorted_pedestrians[1] and h_box_size>=sorted_pedestrians[2]:
                                    cv.rectangle(frame4, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), color, thickness=2)
                                    # put the class name text on the detected object
                                    cv.putText(frame4, 'pedestrian', (int(box2[0]), int(box2[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1)

           
            #join frames together
            h1 = cv.hconcat([frame1, frame2])
            h2 = cv.hconcat([frame3, frame4])
            final_output = cv.vconcat([h1,h2])
            cv.imshow('Task2', final_output)
            
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
        
    vid.read()
    cv.destroyAllWindows()
    
#task2('TownCentreXVID.avi')


#task implementation
def parse_and_run():
    
    if str(sys.argv[1]) == '–b':
        videoFile = str(sys.argv[2])
        task1(videoFile)
    elif str(sys.argv[1]) == '–d':
        videoFile = str(sys.argv[2])
        task2(videoFile)
    else:
        print('Please enter correct arguments!')
        
if __name__ == '__main__':
    parse_and_run()
