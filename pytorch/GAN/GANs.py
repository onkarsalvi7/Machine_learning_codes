# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:26:43 2018

@author: Onkar
"""

from itertools import count
import torchvision
import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as D
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def get_input():
    #getting the training and the test dataset
    train_data =datasets.MNIST('./Data/mnist', train = True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))]))
    
    test_data =datasets.MNIST('./Data/mnist', train = False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))]))
    
    return train_data
#Getting Generator Input



#--------------------------------------------------------------------------


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5, 1, 0, bias = False) #24
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias = False) #12
        self.conv3 = torch.nn.Conv2d(64, 128, 5, 1, 0, bias = False) #8
        self.conv4 = torch.nn.Conv2d(128, 256, 4, 2, 1, bias = False) #4
        self.conv5 = torch.nn.Conv2d(256,  1, 4, 1, 0, bias = False) #1
        self.batchnorm1 = torch.nn.BatchNorm2d(32)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.batchnorm3 = torch.nn.BatchNorm2d(128)
        self.batchnorm4 = torch.nn.BatchNorm2d(256)        

        
    def forward(self,x):
        x = F.elu(self.batchnorm1(self.conv1(x)),0.2) #12
        x = F.elu(self.batchnorm2(self.conv2(x)),0.2) #8
        x = F.elu(self.batchnorm3(self.conv3(x)),0.2) #3
        x = F.elu(self.batchnorm4(self.conv4(x)),0.2) #1
        x = F.elu(self.conv5(x))#1        
        return F.sigmoid(x)

    
#-----------------------------------------------------------------------------

    
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(100,256,4,1,0, bias =  False)
        self.deconv2 = torch.nn.ConvTranspose2d(256,512,4,2,1, bias =  False)
        self.deconv3 = torch.nn.ConvTranspose2d(512,1024,5,1,0, bias =  False)
        self.deconv4 = torch.nn.ConvTranspose2d(1024,2048,4,2,1, bias =  False)
        self.deconv5 = torch.nn.ConvTranspose2d(2048,1,5,1,0, bias =  False)
        self.batchnorm1 = torch.nn.BatchNorm2d(256)
        self.batchnorm2 = torch.nn.BatchNorm2d(512)
        self.batchnorm3 = torch.nn.BatchNorm2d(1024)
        self.batchnorm4 = torch.nn.BatchNorm2d(2048)

        
    def forward(self,x):
        x = F.elu(self.batchnorm1(self.deconv1(x))) #12
        x = F.elu(self.batchnorm2(self.deconv2(x))) #8
        x = F.elu(self.batchnorm3(self.deconv3(x))) #3
        x = F.elu(self.batchnorm4(self.deconv4(x))) #1
        x = self.deconv5(x)#1        
        return F.tanh(x)

#----------------------------------------------------------------------------

gen = Generator()
dis = Discriminator()

if torch.cuda.is_available(): 
    gen = gen.cuda()
    dis = dis.cuda() 
    
optimizerD = optim.SGD(dis.parameters(), lr = 0.005, momentum=0.1) 
optimizerG = optim.SGD(gen.parameters(), lr = 0.005, momentum=0.1) 


#-----------------------------------------------------------------------------

def train(epoch):
    train_data = get_input()
    noise = torch.FloatTensor(64, 100, 1, 1)
    fixed_noise = torch.FloatTensor(64, 100, 1, 1).normal_(0,1)
    label_real = torch.ones(64)
    label_fake = torch.zeros(64)
    label_real = label_real.float()
    label_fake = label_fake.float()
                              
    label = torch.FloatTensor(64)
    image = torch.FloatTensor(64, 1, 28, 28)
    criterion = torch.nn.BCELoss()
    
    if torch.cuda.is_available():
        image = image.cuda()
        label_real, label_fake =  label_real.cuda(), label_fake.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        criterion.cuda()

    
    train_loader = D.DataLoader(train_data, batch_size = 64, drop_last = True)
    for i in range(1, epoch+1):
        for batch_idx,(images, label) in enumerate(train_loader):
            #--------------------------------------Discriminator
            #-------Train with real data----------
            image.copy_(images)
            imagev = Variable(image)
            labelv = Variable(label_real)
            dis.zero_grad()
            output = dis(imagev)
            loss_real = criterion(output, labelv)
            loss_real.backward()
            d_x = output.data.mean()
            
            #--------Train with fake data-----------
            noise.copy_(torch.FloatTensor(64, 100, 1, 1).normal_(0,1))
            noisev = Variable(noise)
            fake = gen(noisev)
            labelv = Variable(label_fake)
            output = dis(fake.detach())
            loss_fake = criterion(output, labelv)
            loss_fake.backward()
            DG_z1 = output.data.mean()
            loss = loss_real + loss_fake
            optimizerD.step()
            
            #------------------------------------Generator
            gen.zero_grad()
            labelv = Variable(label_real)
            output = dis(fake)
            lossG = criterion(output, labelv)
            lossG.backward()
            DG_z2 = output.data.mean()
            optimizerG.step()
    
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (i, epoch, batch_idx, 200,
                     loss_real.data[0], loss_fake.data[0], d_x, DG_z1, DG_z2))
        
            if (batch_idx == 200):
                break
        



        # Save Model
        if i % 5 == 0:
            torch.save(gen.state_dict(), 'netG.pth')
            torch.save(dis.state_dict(), 'netD.pth')
        
    return fake.data
        

        

#getting fake images from the network
fake = train(100) 

#Saving the fake images
torchvision.utils.save_image(fake,'fake_samples.png')
