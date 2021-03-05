from __future__ import print_function
import os
# import nni
import time
import torch
import logging
import argparse
import torchvision
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from birds_get_tree_target_2 import *
import torchvision.transforms as transforms
import torchvision.models as models

logger = logging.getLogger('fine-grained-or-not')
 

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BATCH_SIZE = 64
Hiden_Number = 600
lr = 0.1
nb_epoch = 100
criterion = nn.CrossEntropyLoss()

criterion_NLLLoss = nn.NLLLoss()


#Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset    = torchvision.datasets.ImageFolder(root='/data/Birds/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True)

testset = torchvision.datasets.ImageFolder(root='/data/Birds/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True)


print('==> Building model..')

net =models.resnet50(pretrained=True)

class model_bn(nn.Module):
    def __init__(self, model, feature_size=512,classes_num=200):

        super(model_bn, self).__init__() 

        self.features_2 =  nn.Sequential(*list(model.children())[:-2])

        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2048 * 1 * 1
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            #nn.Linear(feature_size, classes_num),
        )

  

        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size , 13),
            nn.Softmax(1)
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size // 3 * 2, 38),
            nn.Softmax(1)
        )

        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size // 3, 200),
            nn.Softmax(1)
        )



 
    def forward(self, x, targets):


        x = self.features_2(x)   
        x = self.max(x)

        x = x.view(x.size(0), -1)

        x = self.features_1(x) # N * 512

        x_1 =  x[:,  0:Hiden_Number//3]
        x_2 =  x[:,Hiden_Number//3:Hiden_Number// 3 * 2]
        x_3 =  x[:,Hiden_Number// 3 * 2:Hiden_Number]

        # x_1 =  x[:,  0:200]
        # x_2 =  x[:,200:400]
        # x_3 =  x[:,400:600]

        order_input  = torch.cat([x_1, x_2.detach(),x_3.detach()],1)
        family_input = torch.cat([     x_2,x_3.detach()],1)
        species_input = x_3




        
#---------------------------------------------------------------------------------------
        order_targets, family_targets= get_order_family_target(targets)


#---------------------------------------------------------------------------------------
        order_out = self.classifier_1(order_input)
        ce_loss_order = criterion_NLLLoss(torch.log(order_out), order_targets) # 13


#---------------------------------------------------------------------------------------
        family_out = self.classifier_2(family_input)

        ce_loss_family = criterion_NLLLoss(torch.log(family_out), family_targets) # 38

#---------------------------------------------------------------------------------------
        species_out = self.classifier_3(species_input)
        ce_loss_species = criterion_NLLLoss(torch.log(species_out), targets)


#---------------------------------------------------------------------------------------
        ce_loss =  ce_loss_order + ce_loss_family + ce_loss_species 

        return ce_loss, [species_out,targets], [family_out, family_targets],\
                        [order_out, order_targets]


use_cuda = torch.cuda.is_available()



net =model_bn(net, Hiden_Number, 200)

if use_cuda:
    net.classifier_1.cuda()
    net.classifier_2.cuda()
    net.classifier_3.cuda()


    net.features_1.cuda()
    net.features_2.cuda()


    net.classifier_1 = torch.nn.DataParallel(net.classifier_1)
    net.classifier_2 = torch.nn.DataParallel(net.classifier_2)
    net.classifier_3 = torch.nn.DataParallel(net.classifier_3)


    net.features_1 = torch.nn.DataParallel(net.features_1)
    net.features_2 = torch.nn.DataParallel(net.features_2)



    cudnn.benchmark = True


def train(epoch,net, trainloader,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    order_correct = 0
    family_correct = 0
    species_correct = 0


    order_total = 0
    family_total= 0
    species_total= 0

    idx = 0
    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        #out, ce_loss = net(inputs, targets)

        ce_loss,\
        [species_out, species_targets],\
        [family_out, family_targets],\
        [order_out, order_targets] = net(inputs, targets)

        loss = ce_loss


        loss.backward()
        optimizer.step()

        train_loss += loss.item()


        _, order_predicted = torch.max(order_out.data, 1)
        order_total += order_targets.size(0)
        order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

        _, family_predicted = torch.max(family_out.data, 1)
        family_total += family_targets.size(0)
        family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

        _, species_predicted = torch.max(species_out.data, 1)
        species_total += species_targets.size(0)
        species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()




    
    train_order_acc = 100.*order_correct/order_total
    train_family_acc = 100.*family_correct/family_total
    train_species_acc = 100.*species_correct/species_total

    train_loss = train_loss/(idx+1) 
    print('Iteration %d, train_order_acc = %.5f,train_family_acc = %.5f,\
train_species_acc = %.5f, train_loss = %.6f' % \
                          (epoch, train_order_acc,train_family_acc,train_species_acc,train_loss))
    return train_order_acc, train_family_acc,train_species_acc,train_loss

def test(epoch,net,testloader,optimizer):

    net.eval()
    test_loss = 0


    order_correct = 0
    family_correct = 0
    species_correct = 0

    order_total = 0
    family_total= 0
    species_total= 0


    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            #out, ce_loss = net(inputs,targets)

            ce_loss,\
            [species_out, species_targets],\
            [family_out, family_targets],\
            [order_out, order_targets] = net(inputs, targets)

            test_loss += ce_loss.item()

            _, order_predicted = torch.max(order_out.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, family_predicted = torch.max(family_out.data, 1)
            family_total += family_targets.size(0)
            family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

            _, species_predicted = torch.max(species_out.data, 1)
            species_total += species_targets.size(0)
            species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()


    test_order_acc = 100.*order_correct/order_total
    test_family_acc = 100.*family_correct/family_total
    test_species_acc = 100.*species_correct/species_total

    test_loss = test_loss/(idx+1)
    print('Iteration %d, test_order_acc = %.5f,test_family_acc = %.5f,\
test_species_acc = %.5f, test_loss = %.6f' % \
                          (epoch, test_order_acc,test_family_acc,test_species_acc,test_loss))
    return test_order_acc, test_family_acc,test_species_acc




 
def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)


optimizer = optim.SGD([
                        {'params': net.classifier_1.parameters(), 'lr': 0.1},
                        {'params': net.classifier_2.parameters(), 'lr': 0.1},
                        {'params': net.classifier_3.parameters(), 'lr': 0.1},
                        {'params': net.features_1.parameters(),   'lr': 0.1},
                         {'params': net.features_2.parameters(),   'lr': 0.01},
    
                        
                     ], 
                      momentum=0.9, weight_decay=5e-4)

if __name__ == '__main__':
    try:
        # main(params)
        max_val_acc = 0
        for epoch in range(nb_epoch):

            optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch)
            optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch) 
            optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch) 
            optimizer.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch) 
            optimizer.param_groups[4]['lr'] =  cosine_anneal_schedule(epoch) / 10

            train(epoch, net,trainloader,optimizer)
            test_order_acc, test_family_acc,test_species_acc = test(epoch, net,testloader,optimizer)
            if test_species_acc >max_val_acc:
                max_val_acc = test_species_acc
            print("max_val_acc ==", max_val_acc)

    except Exception as exception:
        logger.exception(exception)
        raise
