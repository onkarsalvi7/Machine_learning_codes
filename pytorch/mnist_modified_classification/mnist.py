# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:49:59 2018

@author: Onkar
"""

from itertools import count
import torchvision
import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchfile
import torch.utils.data as D
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

#getting the training and the test dataset
train_data =datasets.MNIST('./Data/mnist', train = True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))

test_data =datasets.MNIST('./Data/mnist', train = False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))

#Getting 10 percent of the dataset
train_loader = D.DataLoader(train_data, batch_size = int(len(train_data)/10), shuffle = True)
test_loader = D.DataLoader(test_data, batch_size = int(len(test_data)/10), shuffle = True)



#x_train0 = torch.Tensor(x_train0)
#x_train1 = torch.Tensor(x_train1)
 #---------Data Pre-Processing----------# 
'''
for batch_id,image in enumerate(train_loader):
    if batch_id==0:
        x_train0 = image
    if batch_id==1:
        x_train1 = image
        break
xtr0 = x_train0[0]
xtr1 = x_train1[0]

ytr0 = x_train0[1]
ytr1 = x_train1[1]
        
        
       
           #combining two images to make a single image
Xtr = torch.cat((xtr0,xtr1),1)


#Adjusting the labels
Ytr = torch.FloatTensor((ytr0.size()))
for i in range(0,len(Ytr)):
    if ytr0[i]==ytr1[i]:
        Ytr[i]=float(1)
    else:
        Ytr[i]=float(0)  
        
'''

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
def process_data(loader):
    
    for i,j in enumerate(loader):
        if i == 0:
            X0 = j
        if i == 1:
            X1 = j
            break
    
    #Loading the images in batches of 10

    xtr0 = X0[0]
    xtr1 = X1[0]
    
    ytr0 = X0[1]
    ytr1 = X1[1]

    
    Xtr0 = torch.cat((xtr0[:int(len(X0[0])*0.7),:,:,:],xtr1[:int(len(X1[0])*0.7),:,:,:]),1)
    Xtr1 = torch.cat((xtr0[int(len(X0[0])*0.7):,:,:,:],xtr0[int(len(X1[0])*0.7):,:,:,:]),1)
    Xtr = torch.cat((Xtr0,Xtr1),0)


    
    
   
       #combining two images to make a single image
    
    print(len(Xtr))
    #Adjusting the labels
    Ytr = torch.ones(ytr0.size())
    for i in range(0,len(Ytr[:int(len(X0[1])*0.7)])):
        if ytr0[i]==ytr1[i]:
            Ytr[i]=float(1)
        else:
            Ytr[i]=float(0)
            
    plt.imshow(Xtr[0,0,:,:])
    plt.show()
    plt.imshow(Xtr[0,1,:,:])
    plt.show()
    print(Ytr[0])
    torchvision.utils.make_grid()
    
    plt.imshow(Xtr[len(Xtr)-1,0,:,:])
    plt.show()
    plt.imshow(Xtr[len(Xtr)-1,1,:,:])
    plt.show()
    print(Ytr[len(Xtr)-1])

            
    #Xtr = D.TensorDataset(Xtr)
    #Ytr = D.TensorDataset(Ytr)

    
    #trainx = D.DataLoader(Xtr, batch_size = 32)
    #trainy = D.DataLoader(Ytr, batch_size = 32)
    return D.TensorDataset(Xtr, Ytr)


#---------Creating a model----------#

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(    2, 8, 5, 1, 0) #24
        self.conv2 = torch.nn.Conv2d(8, 16, 4, 2, 1) #12
        self.conv3 = torch.nn.Conv2d(16, 32, 5, 1, 0) #8
        self.conv4 = torch.nn.Conv2d(32, 64, 4, 2, 1) #4
        self.conv5 = torch.nn.Conv2d(64,    10, 4, 1, 0) #1
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #12
        x = F.relu(self.conv2(x)) #8
        x = F.relu(self.conv3(x)) #3
        x = F.relu(self.conv4(x))#1
        x = F.relu(self.conv5(x))#1        
        x = x.view(-1, 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)

model = ConvNet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

#-----------TRaining the model------------#

def train(trainx, epoch):
    loss_data=[]
    accuracy = []
    for i in range(epoch):
        accuracy = []
        for j,(image,label) in enumerate(trainx):
            '''
            a = iter(trainx)
            b = iter(trainy)
            image = next(a)
            label = next(b)
            '''
            image = Variable(image)
            label = Variable(label.float())    
            optimizer.zero_grad()
            output = model(image)
            loss = F.binary_cross_entropy(output,label)
            loss_ = loss.data[0]
            o=output.data.numpy()
            accuracy_ = len(o[o==label.data.numpy()])/len(o)
            accuracy.append(accuracy_)
            loss_data.append(loss_)
            loss.backward()
            optimizer.step()
            print(loss_)

    return loss_data, accuracy  

       
def test(testx):
    loss_data=[]
    accuracy= []
    for i,(image,label) in enumerate(testx):
        '''
        a = iter(testx)
        b = iter(testy)
        image = next(a)
        label = next(b)
        '''            
        image = Variable(image)
        label = Variable(label.float())
        optimizer.zero_grad()
        output = model(image)
        loss = F.binary_cross_entropy(output,label)
        loss_ = loss.data[0]
        o=output.data.numpy()
        accuracy_ = len(o[o==label.data.numpy()])/len(o)
        accuracy.append(accuracy_)
        loss_data.append(loss_)
        print(loss_)
    return loss_data, accuracy 
        
 

#--------Running the model----------#

train_set = process_data(train_loader)
test_set = process_data(test_loader)
'''
trainx = D.DataLoader(Xtr, batch_size = 32)
trainy = D.DataLoader(Ytr, batch_size = 32)
'''
trainx = D.DataLoader(train_set, batch_size = 32, drop_last = True, shuffle = True)
testx = D.DataLoader(test_set, batch_size = 32, drop_last = True, shuffle = True)

train_loss, train_accuracy = train(trainx, 100)

test_loss, test_accuracy = test(testx)

plt.plot(range(len(train_loss)), train_loss)
loss_test = np.mean(test_loss)

trainx = D.DataLoader(train_set, batch_size = len(train_set), drop_last = True, shuffle = True)
testx = D.DataLoader(test_set, batch_size = len(test_set), drop_last = True, shuffle = True)

















