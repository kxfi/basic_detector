"""
Detector Manager is the base class that will be used to control the platform. 

"""



# Import the necessary packages
import os
import sys
import time
import logging
import argparse
import threading
import multiprocessing
import numpy as np
import cv2
import imutils 
import socket
import json
#import serial
import random
import torch
from threading import Thread
import datetime


#os.chdir("C:\\Users\\Ken\\Desktop\\baby_Talos\\TALOS_APP")
try:
    from .vision_manager import Vision_Manager
except:
    from vision_manager import Vision_Manager

class BasicDetector(Vision_Manager):
    def __init__(self,config):
        super().__init__(self)
        self.config = config
        # PATH IS WHAT CHANGES! IT IS A .PT FILE
        self.model_path = "/Volumes/T7 Touch/Jeremy_MINES/Jeremy_CODE/Models/minenet2.pt"# '/home/atr/Talos_spot/App/MODELS/groundnet_atr.pt'
        self.detection_data = {"Front":[]}

    def start(self):
        try:
            self.thread_model = Thread(target = self.load_model) # Check constant connection to an IP address
            self.thread_model.start()
            # Warm up model
        except:
            self.MODEL_READY = False
            print("Model Not working")


    def load_model(self):
        # load model 
        self.model = self.get_object_detector()
        # Warm up model
        self.warm_up_model(self.model,img_size=(self.img_height, self.img_width, 3))

    def get_object_detector(self):
        """ Make a model object from the trained algorithm """
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        # Custom model
        model = torch.hub.load("ultralytics/yolov5","custom",path = self.model_path) #force_reload=True) # you will need this uncommented line if you get errors
        # Yolov5 model
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
        model.conf = 0.4
        # Threshhold 
        model.iou = 0.5
        return model


    def warm_up_model(self,model, img_size):
        # warm up model to start up better
        img = np.zeros(img_size, dtype="uint8")
        print(f"{img.shape}")
        print("Warming up model")
        for i in range(1, 2):
            model(img)
        print(f"Model Ready")
        self.MODEL_READY = True


    def detect(self,frame):
        
        dets = []
        results = self.model(frame)

        # get detections 
        detection_init = results.pred

        detection = np.vstack(detection_init)
        #print("Detectionss: ",detection)


        # loop through detections
        if detection.size == 0:
            self.detections = []

        else:
            for i in detection:
                # Have a true append for each detection
                self.APPEND = True 
                print(i)

                # start bounding box logic
                x = int(i[0])
                y = int(i[1])
                x_max = int(i[2])
                y_max = int(i[3])
                confidence = round(float(i[4]),2)
                id = int(i[5])

                # Dimensions 
                w = x_max - x
                h = y_max - y

                # Give the detection names 
                '''
                id = int(id)

                if id == 0:
                    name = "person"
                elif id == 1:
                    name = "vehicle"
                elif id == 2:
                    name = "helicopter"
                elif id == 3:
                    name = "tank_whl"
                elif id == 4:
                    name = "tank_trk"
                elif id == 5:
                    name = "BMP2"
                elif id == 99:
                    name = "object"
                else:
                    name = "mil vehicle"
                '''
                id = int(id)
                
                if id == 0:
                    name = "person"
                elif id == 1:
                    name = "car"
                elif id == 2:
                    name = "helicopter"
                elif id == 3:
                    name = "BTR"
                elif id == 4:
                    name = "T72"
                elif id == 5:
                    name = "BMP2"
                elif id == 6:
                    name = "ZSU234"
                elif id == 7:
                    name = "SA-13"
                elif id == 8:
                    name = "2S1"
                elif id == 9:
                    name = "SA-10"
                elif id == 10:
                    name = "M113A2"
                elif id == 11:
                    name - "Long_Track"
                elif id == 12:
                    name = "HMMWV"
                elif id == 13:
                    name = "SA-6"
                elif id == 14:
                    name = "MTLB"
                elif id == 15:
                    name = "SA-9"
                elif id == 16:
                    name = "SA-8"
                elif id == 17:
                    name = "2S19"
                elif id == 18:
                    name = "ZPU1"
                elif id == 19:
                    name = "SA-22"
                elif id == 20:
                    name = "tank"
                elif id == 21:
                    name = "Abrams"
                elif id == 22:
                    name = "PTKM"
                elif id == 23:
                    name = "AT"
                elif id == 24:
                    name = "EFP"
                elif id == 25:
                    name = "Artillery"
                elif id == 26:
                    name = "AP"
                
            
                timestamp = datetime.datetime.now().timestamp()

                # If append is true 
                if self.APPEND == True:
                    dets.append(
                        {
                            "name": name,
                            "conf": confidence,
                            "time": timestamp,
                            "box" :
                            {
                                "x":    x,
                                "y":    y,
                                "wid":  w,
                                "hei":  h
                            }})
                print("Dets: ",dets)


            self.detections = dets

        return self.detections

    def get_data(self,frame):
        # Get the signal strength and if connected
        self.connect, self.signal_strength = self.get_signal_strength()
        # Get the sensor data
        if self.MODEL_READY == True:
            self.detection_data = self.detect(frame)
            self.detection_data = {"Front":self.detection_data}
            print(self.detection_data)

        # If there is no sensor data return 0,0,0
        # if self.detection_data == None:
        #     return self.connect,self.signal_strength,None

        return self.connect, self.signal_strength, self.detection_data

# def get_distance(frame,detection_data, car_location):
#     # Find the bounding boxes for the car
#     # for detection in frame[0]['box']:


#     #     left = int(detection[0])
#     #     top = int(detection[1])
#     #     right = int(detection[2])
#     #     bottom = int(detection[3])

#         dets = []
#         x_mins = []
#         #print("xs:",x_mins)
#         y_mins = []
#         x_maxs = []
#         y_maxs = []
#         confidence = []
#         confs = []
#         obj_classes = []

#         for i in range(len(detection_data)):
#             # time = detection_data[i]["time"]
#             clas = detection_data[i]["name"]
#             x = detection_data[i]["box"]["x"]
#             y = detection_data[i]["box"]["y"]
#             w = detection_data[i]["box"]["wid"]
#             h = detection_data[i]["box"]["hei"]
#             conf = detection_data[i]["conf"]
#             # x_mins.append(int(x))
#             # y_mins.append(int(y))
#             # x_maxs.append(int(x+w))
#             # y_maxs.append(int(y+h))
#             left = int(x)
#             top = int(y)
#             right = int(x+w)
#             bottom = int(y+h)

#         # Draw the bounding box on the frame
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Find the center of the bounding box
#         cx = (left + right) // 2
#         cy = (top + bottom) // 2

#         # Draw a circle at the center of the bounding box
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#         if car_location is None:
#             car_location = (cx, cy)
#         else:
#             # Calculate the distance from the previous car location
#             dx = cx - car_location[0]
#             dy = cy - car_location[1]
#             distance = ((dx ** 2) + (dy ** 2)) ** 0.5

#             # Print the distance
#             print("Distance:", distance)

#             # Update the car location
#             car_location = (cx, cy)

#         return car_location



if __name__ == "__main__":
    # Create a video capture object to read videos
    cam_ip = 1 #"rtsp://admin:Superstarboi321!@192.168.0.175/1"
    # Replace url to test with video
    cam = "/Volumes/T7 Touch/Jeremy_MINES/MINES/IR_mines/VIDS/top_video.mp4" 
    cap = cv2.VideoCapture(cam)

    config = 1
    
    # Create a detector objectDetector_Manager

    detector = BasicDetector(config)
    detector.start()

    limit = 10000000
    count = 0
    car_location=None

    while limit > count:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Use the detector to find the objects
        connected,ss,detections = detector.get_data(frame)
        print(detections)

        # # 
        # # Draw the detections on the frame
        frame = detector.show_detection(frame,detections["Front"])
        count += 1
        # Display the resulting frame
        cv2.imshow('frame',frame)
        
        cv2.waitKey(10)

    detector.end()
    cv2.destroyAllWindows()
    cap.release()
