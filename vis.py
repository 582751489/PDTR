# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:26:02 2020

@author: 58275
"""
import os
import argparse
import cv2
import matplotlib.pyplot as plt 
import numpy as np

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--C1_index',default='12', type=str, help='test_image_index')
parser.add_argument('--C2_index',default='5', type=str, help='test_image_index')
parser.add_argument('--C3_index',default='11', type=str, help='test_image_index')
parser.add_argument('--img_dir',default='Z:\\pro2\\whole\\person',type=str, help='./test_data')
parser.add_argument('--mov_dir',default='Z:\\pro2\\whole\\output',type=str, help='./test_data')


opts = parser.parse_args()

def visualization(index,cam,mov):
    mov_path = mov +'\\output%s'%cam[1]+'.avi'
    cap = cv2.VideoCapture(mov_path)
    rate = cap.get(5)
    FrameNumber = cap.get(7)
    duration = FrameNumber/rate
    indexid = (int(cam[1])-1)*200+int(index)
    root_path = opts.img_dir+'\\gallery\\'+'%04d'%indexid
    ids = os.listdir(root_path)#eg:0002_c1_0001_0.08.jpg
    x = []
    for i in ids:
        x.append(float(i[13:17]))
    return duration,x

def vis1():
    duration1,x1 = visualization(opts.C1_index,'c1',opts.mov_dir)
    duration2,x2 = visualization(opts.C2_index,'c2',opts.mov_dir)
    duration3,x3 = visualization(opts.C3_index,'c3',opts.mov_dir)
    p1 = plt.barh('C1', left=x1[0], height=0.5, width=x1[-1]-x1[0],linewidth = duration1,label = 'C1-ID:%s'%opts.C1_index)
    p1 = plt.barh('C2', left=x2[0], height=0.5, width=x2[-1]-x2[0],linewidth = duration2,label = 'C2-ID:%s'%opts.C2_index)
    p1 = plt.barh('C3', left=x3[0], height=0.5, width=x3[-1]-x3[0],linewidth = duration3,label = 'C3-ID:%s'%opts.C3_index)
    plt.title('Time Reference')
    plt.xlabel("time(s)")
    plt.ylabel('Camera')
    plt.legend()
    plt.savefig("timetable.png")
    #plt.show()

def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)


def show(dirs,index):
    for i in range(3):
        query_path = dirs+'\\gallery\\'+'%04d'%(int(index)+400)
        img =os.listdir(query_path)
        indeximg = img[int(len(img)/(i+2))]
        img_path = query_path+'\\'+indeximg    
        ax = plt.subplot(1,11,i+7)
        ax.axis('off')
        imshow(img_path)
        ax.set_title('Camera3'+'_ID%s'%index,color='green')
def vis2():
    fig = plt.figure(figsize=(20,5))
    show(opts.img_dir,opts.C1_index)
    show(opts.img_dir,opts.C2_index)
    show(opts.img_dir,opts.C3_index)
    fig.savefig("pesontable.png")

if __name__ == '__main__':
    vis1()
    vis2()
    