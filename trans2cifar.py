# -*- coding: utf-8 -*-
import numpy as np
import chardet
from PIL import Image
import operator
from os import listdir
import sys
import pickle
import random
  
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin-1')
    return dict
#cc=unpickle("./dataset/cifar-10/cifar-10-batches-py/data_batch_1")
#print(cc)
  
data={}
list1=[]
list2=[]
list3=[]
#将图片转化为32*32的三通道图片
def img_tra():
    for k in range(0,num):
        currentpath=folder+"/"+imglist[k]
        im=Image.open(currentpath)
        #width=im.size[0]
        #height=im.size[1]
        x_s=32
        y_s=32
        #out = im.resize((x_s,y_s),Image.ANTIALIAS)
        out = im.resize((x_s,y_s),Image.LANCZOS)
        out.save(folder_ad+"/"+str(imglist[k]))
  
def addWord(theIndex,word,adder):
    theIndex.setdefault(word,[]).append(adder)
def seplabel(fname):
    filestr=fname.split(".")[0]
    label=int(filestr.split("_")[0]) #图片的命名 _前面是类别
    return label
def mkcf():
    global data
    global list1
    global list2
    global list3
    for k in range(0,num):
        currentpath=folder_ad+"/"+imglist[k]
        im=Image.open(currentpath)
        with open(binpath, 'a') as f:
            for i in range (0,32):
                for j in range (0,32):
                    cl=im.getpixel((i,j))
                    list1.append(cl[0])  #R
  
            for i in range (0,32):
                for j in range (0,32):
                    cl=im.getpixel((i,j))
                    #with open(binpath, 'a') as f:
                    #mid=str(cl[1])
                    #f.write(mid)
                    list1.append(cl[1]) #G
  
            for i in range (0,32):
                for j in range (0,32):
                    cl=im.getpixel((i,j))
                    list1.append(cl[2]) ##B
        list2.append(list1)
        list1=[]
        f.close()
        print("image"+str(k+1)+"saved.")
        list3.append(imglist[k])    #name of pictures
    arr2=np.array(list2,dtype=np.uint8)
    data['batch_label']='training batch 5 of 5' #training batch 1 of 5 testing batch 1 of 1
    data.setdefault('labels',label)
    data.setdefault('data',arr2)
    data.setdefault('filenames',list3)
    output = open(binpath, 'wb')
    pickle.dump(data, output)
    output.close()
  
folder="./new_bxhz/hz_4"  #自己图片的路径 train_batch_5 test
folder_ad="./train_batch_7_ad" #将图片转化为32*32的三通道图片的路径  train_batch_5_ad test_ad
imglist=listdir(folder) #这里原作者好像写错了，我自行修改了，目测现在是对的
num=len(imglist)
img_tra()
label=[]
for i in range (0,num):
    label.append(seplabel(imglist[i]))
binpath="./new_dataset/data_batch_8" #保存的路径 data_batch_5 test_batch
print(binpath)
mkcf()
