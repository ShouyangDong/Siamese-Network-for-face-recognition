#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:46:03 2018

@author: dongshouyang
"""
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps 
import mxnet.ndarray as nd
from mxnet.gluon import nn
from mxnet import autograd

def imshow(img,text,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  
    
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
class Config():
    training_dir = "/Users/dongshouyang/jupyternotebook/cv/face_recognition/training/"
    testing_dir = "/Users/dongshouyang/jupyternotebook/cv/face_recognition/validation/"
    train_batch_size = 90
    train_number_epochs = 100
    
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=True,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset)

        img0 = (img0_tuple[0]).asnumpy()
        img1 = (img1_tuple[0]).asnumpy()
        img0= Image.fromarray(img0)
        img1= Image.fromarray(img1)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img0 = np.asarray(img0,dtype=np.float32)
        img1 = np.asarray(img1,dtype=np.float32)
        img0 = img0.reshape((1, 112, 92))
        img1 = img1.reshape((1, 112, 92))
            
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform:
            img0 = mx.image.imresize(img0, 100, 100)
            img1 = mx.image.imresize(img1, 100, 100)
        return img0, img1 , nd.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)
    
    def __len__(self):
        return len(self.imageFolderDataset)
    
    
folder_dataset = ImageFolderDataset(root=Config.training_dir)   
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=False ,should_invert=False)

class SiameseNetwork(nn.Block):  
    def __init__(self, verbose=False,**kwargs):
        super(SiameseNetwork, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outer most Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Sequential()
            b1.add(
                nn.Conv2D(4, kernel_size=3, padding=1, activation='relu'),
                nn.BatchNorm(),
                nn.Dropout(0.2)
            )
            
            b2 = nn.Sequential()
            b2.add(
                nn.Conv2D(8, kernel_size=3, padding=1, activation='relu'),
                nn.BatchNorm(),
                nn.Dropout(0.2)
            )
        
            b3 = nn.Sequential()
            b3.add(
                nn.Conv2D(8, kernel_size=3, padding=1, activation='relu'),
                nn.BatchNorm(),
                nn.Dropout(0.2)
            )
       
            b4 = nn.Sequential()
            b4.add(
                nn.Flatten(),
                nn.Dense(500, activation="relu"),
                nn.Dense(500, activation="relu"),
                nn.Dense(5)
            )
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4)
        

    def forward(self, input1, input2):
        output1 = self.net(input1)
        output2 = self.net(input2)
        return output1,output2
    
class ContrastiveLoss(nn.Block):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nd.sqrt(nd.sum(nd.power(nd.subtract(output1, output2),2))) 
        loss_contrastive = nd.mean(nd.add(nd.subtract(1,label) * nd.power(euclidean_distance, 2),(label) * nd.power(nd.subtract(self.margin, euclidean_distance), 2)))
        return loss_contrastive

train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8,batch_size=Config.train_batch_size)
from mxnet import init
net = SiameseNetwork()
criterion = ContrastiveLoss()
batch_size=Config.train_batch_size


ctx = mx.cpu(0)
counter = []
loss_history = [] 
iteration_number= 0

net.initialize(init.Xavier(magnitude=2), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate': 0.005, 'wd': 0.001})

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0 = img0.as_in_context(ctx)
        img1 = img1.as_in_context(ctx)
        label= label.as_in_context(ctx)
        with autograd.record():
            output1,output2 = net(img0, img1)
            loss_contrastive = criterion(output1,output2,label)   
        loss_contrastive.backward()        
        trainer.step(batch_size)
        if i %10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.asnumpy()[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.asnumpy()[0])
show_plot(counter,loss_history) 