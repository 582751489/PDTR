#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
import copy
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
import PIL.ImageOps
import shutil
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/test.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
ap.add_argument("-camera", "--camera",help="camera1 or 2 or 3", default = "c1")
ap.add_argument("-ids", "--ids",help="index ids ",type=str, default = False)
ap.add_argument("--out",help="out put ",type=list, default = './')
args = vars(ap.parse_args())
#print(args["camera"])
pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
#list = [[] for _ in range(100)]



def main(yolo):
    t0 = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)#跟踪使用

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(args["input"])


    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if args["ids"] == False:  
            out = cv2.VideoWriter('./output/output%s.avi'%args["camera"][1], fourcc, 50, (w, h))
        else:
            out = cv2.VideoWriter('./output/output%s_reid.avi'%args["camera"][1], fourcc, 50, (w, h))
        list_file = open('detection_rslt.txt', 'w')
        frame_index = -1
        nump=1
    #fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
        frame2=copy.deepcopy(frame)
        #image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb 仅yolo使用
        boxs, confidence, class_names = yolo.detect_image(image)
        print(boxs)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        makequery = True
        for det in detections:
            bbox = det.to_tlbr()

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)#跟踪框
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            list_file.write(str(frame_index)+',')#3-5-7-9
            list_file.write(str(track.track_id)+',')#画面内的所有人id
            b0 = str(bbox[0])#.split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1])#.split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2]-bbox[0])#.split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3]-bbox[1])

            #放置id
            list_file.write(str(b0) + ','+str(b1) + ','+str(b2) + ','+str(b3))
            list_file.write('\n')
            if len(class_names) > 0: 
                class_name = class_names[0]
                cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)#person
 
            i += 1
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
 
            pts[track.track_id].append(center)
 
            thickness1 = 5
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(255,255,255),thickness)
            if args["ids"] == False:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
                cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)#id
                cv2.circle(frame,  (center), 1, color, thickness1)
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                       continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                try:
                    num = (int(args["camera"][1])-1)*200
                    path = 'Z:\\pro2\\whole\\person\\gallery\\%04d'%int(track.track_id+num)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    if len(os.listdir(path)) <= 150: #最多存储150张相片
                        crop = frame2[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                        crop = cv2.resize(crop,(64,128),interpolation=cv2.INTER_AREA)#CUBIC 对扩大图片 area 对缩小图片
                        filepath = path +'\\'+'%04d'%int(track.track_id+num)+'_%s_'%args["camera"]+'%04d'%int(len(os.listdir(path))+1)+'_%.2f'%(video_capture.get(0)/1000) +'.jpg'  #%04d
                        cv2.imwrite(filepath,crop)
                except:
                    continue
                
            #单独索引
            else:
                makequery = False
                id1 = int(args["ids"])
                if int(track.track_id) == id1 :
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
                    cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)#id
                    cv2.circle(frame,  (center), 1, color, thickness1)
                    for j in range(1, len(pts[track.track_id])):
                        if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                           continue
                        thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                        cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                    cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)#person
                else:
                    continue
        count = len(set(counter))
        cv2.putText(frame, "Total Pedestrian Counter: "+str(count),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Pedestrian Counter: "+str(i),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),2)
        cv2.namedWindow("YOLO4_Deep_SORT", 0);
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768);
        cv2.imshow('YOLO4_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #makequery
    if makequery == True:
        root_path = 'Z:\\pro2\\whole\\person\\gallery\\'
        copy_path = 'Z:\\pro2\\whole\\person\\query\\'
        ids = os.listdir(root_path)
        #print(ids)
        for i in ids:
            img_path = root_path + i
            img =os.listdir(img_path)
            indeximg = img[int(len(img)/2)]
            old_name = img_path+'\\'+indeximg
            new_path = copy_path + i
            new_name = new_path + '\\' + indeximg
            if not os.path.exists(new_path):
                os.makedirs(new_path) 
            shutil.copyfile(old_name, new_name)
    print(" ")
    print("[Finish]")
    end = time.time()
    print("the whole time ",end - t0)
    if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main(YOLO())
