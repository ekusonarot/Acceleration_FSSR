from typing import FrozenSet
from matplotlib.pyplot import cla
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
from torchvision.transforms.transforms import CenterCrop
import numpy as np
import dataset
import classification
import models
import torch
from torch import optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batchsize = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
fsrcnn = models.FSRCNN(4, 3).to(device)
light_weight_fsrcnn = models.FSRCNN(4, 3, d=32, m=1).to(device)
fsrcnn.load_state_dict(torch.load("./weights/x4_3ch_fsrcnn_9.pth", map_location=torch.device(device)))
for param in fsrcnn.parameters():
    param.requires_grad = False
light_weight_fsrcnn.load_state_dict(torch.load("./weights/x4_3ch_lightfsrcnn_9.pth", map_location=torch.device(device)))
for param in light_weight_fsrcnn.parameters():
    param.requires_grad = False

classify = classification.CategoricalCNN(light_weight_fsrcnn, fsrcnn, input_shape=(3, 180, 320), device=device).to(device)
#classify.load_state_dict(torch.load("./weight_classification_0.pth", map_location=torch.device(device)), strict=False)
optimizer = optim.Adam([
    {'params': classify.class_first_part.parameters(), "lr": 1e-3},
    {'params': classify.class_second_part.parameters(), "lr": 1e-4},
    {'params': classify.class_last_part.parameters(), 'lr': 1e-7}
    ])

transform_input = transforms.Compose([
    #transforms.Pad(padding=(0,0,16,0), fill=0, padding_mode='constant'),
    transforms.Resize((180,320)),
    transforms.ToTensor()
    ])
transform_target = transforms.Compose([
    #transforms.Pad(padding=(0,0,16,0), fill=0, padding_mode='constant'),
    transforms.ToTensor()
])
dataset = dataset.MyDataset("./train_sharp", transform_input, transform_target)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

classify.train()
for i in range(10):
    print("epoch{} start".format(i))
    j = 0
    
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs, class_vector = classify(inputs)
        loss = classify.Loss(outputs, targets, class_vector)
        loss.backward()
        optimizer.step()
        j+=1
        #img = outputs[0,:,:,:].to('cpu')
        #plt.imshow(img.permute(1,2,0))
        #plt.show()
        #img = targets[0,:,:,:].to('cpu')
        #plt.imshow(img.permute(1,2,0))
        #plt.show()
        if j%10 == 0:
            print("epoch{}: {}/{}: loss: {}".format(i,j*batchsize,dataloader.__len__()*batchsize,loss))
            print(class_vector[:8])
            print(torch.var(class_vector))
            print(sum(class_vector>0.5))
            print(sum(class_vector<=0.5))
        del loss
        torch.save(fsrcnn.state_dict(), "./test.pth")
    torch.save(fsrcnn.state_dict(), "./weight_classification_{}.pth".format(i))