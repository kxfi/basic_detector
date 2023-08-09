# Imports
import json
import numpy as np
import sys
import os
import time
import logging
import datetime
import random
import time 
import csv 
import threading
import cv2
import ast



# Vision
class Vision_Manager:
    """Sim that gives static data """
    def __init__(self,config):
        # Open and add Config file
        self.config = config

        self.connect = False
        self.signal_strength = 0

                # Boolens
        self.RUN = True

        self.MODEL_READY=False
        self.model_path = "/Users/jeremyacheampong/Developer/CounterMine/Jeremy_CODE/exp7/weights/minenet2.pt"


        # Models
        self.color_model_path = 'MODELS/colornet.pt'
        self.ground_model_path = 'MODELS/groundnet_atr.pt'
        self.air_model_path = 'MODELS/skynet_proto.pt'

        # model params
        # self.threshold = self.config["model"]["threshold"] #= 0.1
        # self.iou = self.config["model"]["iou"] # = 0.45 
        # self.model_kind = self.config["model"]["model_type"] #= "ultralytics/yolov5"
        # self.model_type = self.config["model"][ "custom"] #= "custom"
        # self.number_of_detection_limit = self.config["model"]["number_of_detection_limit"] 
        
        # Camera params
        self.img_height = 480
        self.img_width = 1920
        self.detection_data = None

        #self.start()

    def start(self):
        print("Detector Started")
        self.connect = True
        self.conf_1 = .91
        self.x1 = 208 # 636
        self.y1 = 233 # 148
        self.wid = 238 # 206
        self.hei = 109 # 156

        self.conf_2 = .65
        self.x2 = 203
        self.y2 = 148
        self.wid2 = 143
        self.hei2 = 84

        self.conf_3 = .45
        self.x3 = 150
        self.y3 = 120
        self.wid3 = 20
        self.hei3 = 15

        self.conf_4 = .92
        self.x4 = 48
        self.y4 = 545
        self.wid4 = 70
        self.hei4 = 23

        self.conf_5 = .81
        self.x5 = 430
        self.y5 = 549
        self.wid5 = 35
        self.hei5 = 78

        self.detections = None
        self.MODEL_READY = True

        # try:
        #     self.thread_model = threading.Thread(target = self.load_model) # Check constant connection to an IP address
        #     self.thread_model.start()
        #     # Warm up model
        # except:
        #     self.MODEL_READY = False
        #     print("Model Not working")
    
    def load_model(self):
        # load model 
        print("Loading Fake Model")
        self.model = self.get_object_detector()

        self.warm_up_model(self.model,imgSize=(self.img_height, self.img_width, 3)) 

    def get_object_detector(self):
        """ Make a model object from the trained algorithm """
        print("Getting fake detector")
        model = 1
        return model

    def warm_up_model(self,model, imgSize):
        # warm up model to start up better
        img = np.zeros(imgSize, dtype="uint8")
        print(f"{img.shape}")
        #print("Warming up model")
        for i in range(1, 10):
            print(f"Warming up model {i}")
        print(f"Model Ready")
        self.MODEL_READY = True

    def detect(self,frame):
        self.front_detections = []
        self.right_detections = []
        self.rear_detections = []
        self.left_detections = []

        conf_1 = self.conf_1 + random.random() * .1 - .05
        x1 = int(self.x1 + random.random() * 10) -  2
        y1 = int(self.y1 + random.random() * 4) -  2
        wid = int(self.wid + random.random() * 4) -  2
        hei = int(self.hei + random.random() * 4) -  2

        conf_2 = self.conf_2 + random.random() * .1 - .05
        x2 = int(self.x2 + random.random() * 10) -  2
        y2 = int(self.y2 + random.random() * 4) -  2
        wid2 = int(self.wid2 + random.random() * 4) -  2
        hei2 = int(self.hei2 + random.random() * 4) -  2

        conf_3 = self.conf_3 + random.random() * .1 - .05
        x3 = int(self.x3 + random.random() * 18) -  9
        y3 = int(self.y3 + random.random() * 13) -  7
        wid3 = int(self.wid3 + random.random() * 10) -  5
        hei3 = int(self.hei3 + random.random() * 10) -  5

        conf_4 = self.conf_4 + random.random() * .1 - .05
        x4 = int(self.x4 + random.random() * 18) -  9
        y4 = int(self.y4 + random.random() * 13) -  7
        wid4 = int(self.wid4 + random.random() * 10) -  5
        hei4 = int(self.hei4 + random.random() * 10) -  5

        conf_5 = self.conf_5 + random.random() * .1 - .05
        x5 = int(self.x5 + random.random() * 18) -  9
        y5 = int(self.y5 + random.random() * 13) -  7
        wid5 = int(self.wid5 + random.random() * 10) -  5
        hei5 = int(self.hei5 + random.random() * 10) -  5


        # Fake detections 
        detection_1 = {"name":"mil_vehicle","conf":conf_1,"sensor":"Front","box":{"x":x1,"y":y1,"wid":wid,"hei":hei}}
        detection_2 = {"name":"person","conf":conf_2,"sensor":"Right","box":{"x":x2,"y":y2,"wid":wid2,"hei":hei2}}
        detection_3 = {"name":"mil_vehicle","conf":conf_3,"sensor":"Rear","box":{"x":x3,"y":y3,"wid":wid3,"hei":hei3}}
        detection_4 = {"name":"person","conf":conf_4,"sensor":"Left","box":{"x":x4,"y":y4,"wid":wid4,"hei":hei4}}
        detection_5 = {"name":"mil_vehicle","conf":conf_5,"sensor":"Left","box":{"x":x5,"y":y5,"wid":wid5,"hei":hei5}}


        if random.random() > .5:
            self.front_detections.append(detection_1)
            if random.random() > .5:
                self.right_detections.append(detection_2)

        if random.random() > .5:
            self.rear_detections.append(detection_3)
        
        if random.random() > .5:
            self.left_detections.append(detection_4)
            if random.random() > .5:
                self.left_detections.append(detection_5)

        self.detections = {"Front":self.front_detections,"Right":self.right_detections,"Rear":self.rear_detections,"Left":self.left_detections} # 
        return self.detections

    def get_detections(self,frame):
        self.detections = self.detect(frame)
        # if self.detections == [] or self.detections == 0:
        #     self.detections = None
        # print(self.detections)
        return self.detections

    def show_detection(self,frame,Data):

        if Data == None or Data == []:
            frame = frame
        else:
            for data in Data:
                one = 1
                #if data['name'] != "soldier":
                if one != 1:
                    pass
                else:
                    #print(Data)
                    #data = data[0]
                    print(data)
                    x1 = int(data['box']['x'])
                    y1 = int(data['box']['y'])
                    x2 = int(data['box']['x']) + int(data['box']['wid'])
                    y2 = int(data['box']['y']) + int(data['box']['hei'])
                    name = data['name']
                    x_cen = x1 + int(data['box']['wid'] / 2)
                    y_cen = y1 + int(data['box']['hei'] / 2)
                    confidence  = data['conf']
                    #draw a rectangle around the object

                    cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0, 255, 0), 3)
                    cv2.putText(frame,str(name) ,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),3) # + "_" +str(confidence)
                    cv2.circle(frame, (x_cen,y_cen), radius=0, color=(0, 255, 0), thickness=3)


        return frame #// dont need to return frame 

    def get_data(self,frame):
        # Get the signal strength and if connected
        self.connect, self.signal_strength = self.get_signal_strength()
        # Get the sensor data
        if self.MODEL_READY == True:
            self.detection_data = self.get_detections(frame)

        # If there is no sensor data return 0,0,0
        # if self.detection_data == None:
        #     return self.connect,self.signal_strength,None

        return self.connect, self.signal_strength, self.detection_data

    def stop(self):
        pass

    def get_signal_strength(self):
        return True, 100

    def end(self):
        print("Ending the Detection thread")
        #self.thread_model.join()

