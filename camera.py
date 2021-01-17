
import numpy as np 
import argparse
import imutils 
import time
import cv2
import os
import path






class VideoCamera(object):
    def __init__(self,input_video , output_video):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        ####################################################
    
        #self.labelsPath ="classes.names"
        #self.LABELS = open(self.labelsPath).read().strip().split("\n")
        self.classesFile = "classes.names"
        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        np.random.seed(42)        
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3),dtype="uint8")
        # derive the paths to the YOLO weights and model configuration
        self.modelWeights="test04/yolov4_best.weights"
        self.modelConfiguration = "test04/yolov4.cfg"

        print("WAIT Running the yolo model...")
        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        #self.ln = self.net.getLayerNames()
        #self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.vs = cv2.VideoCapture(input_video)
        self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.confThreshold = 0.1
        self.nmsThreshold = 0.4
        self.inpWidth = 412
        self.inpHeight = 412
        self.outputFrame = None
        #self.outputFile = 'static/uploads/'+output_video[:-4]+"_pred.mp4"
        self.outputFile = output_video
        print("outputFile :"+self.outputFile)
        #self.vid_writer = cv2.VideoWriter(self.outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH)),round(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        self.vid_writer = cv2.VideoWriter(self.outputFile, fourcc, 30, (round(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH)),round(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        #self.starting_time = time.time()
        self.total = 0
        self.starting_time = time.time()
        # FPS = 1/X
        # # X = desired FPS
        # self.FPS = 500
        # self.FPS_MS = int(self.FPS * 1000)

        #self.vs= cv2.VideoCapture(0)
        (self.W, self.H) = (None, None)
        

   

    def __del__(self):
        self.vs.release()

    # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self ,frame , classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
        
        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)
    
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            
    def postprocess(self ,frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
    
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        nb = 0
        for i in indices:
            nb = nb + 1
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame ,classIds[i], confidences[i], left, top, left + width, top + height)
        return nb    


    def get_frame(self):

        hasFrame, frame = self.vs.read()
        self.total += 1  
        try :  
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        except :
            return -1

        self.net.setInput(blob)
        #outs = self.net.forward(self.getOutputsNames(self.net))
        outs = self.net.forward(self.getOutputsNames())

        nb = self.postprocess(frame, outs)

        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        elapsed_time = time.time() - self.starting_time
        fps = self.total / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.putText(frame, "Number of Potholes : " + str(round(nb, 2)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0) , 2)
        self.vid_writer.write(frame.astype(np.uint8))

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()