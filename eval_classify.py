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
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)
fsrcnn = models.FSRCNN(4, 3).to(device)
light_weight_fsrcnn = models.FSRCNN(4, 3, d=32, m=1).to(device)
fsrcnn.load_state_dict(torch.load("./weights/x4_3ch_fsrcnn_9.pth", map_location=torch.device(device)))
light_weight_fsrcnn.load_state_dict(torch.load("./weights/x4_3ch_lightfsrcnn_9.pth", map_location=torch.device(device)))

classify = classification.CategoricalCNN(light_weight_fsrcnn, fsrcnn, input_shape=(3, 180, 320), device=device).to(device)
classify.load_state_dict(torch.load("./weight_classification_3.pth", map_location=torch.device(device)), strict=False)

transform_input = transforms.Compose([
    transforms.Resize((180,320)),
    transforms.ToTensor()
    ])
transform_target = transforms.Compose([
    transforms.ToTensor()
])
dataset = dataset.MyDataset("./train_sharp", transform_input, transform_target)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

classify.eval()
for inputs, targets in dataloader:

    inputs = inputs.to(device)
    targets = targets.to(device)
    #with torch.no_grad():
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        outputs, _ = classify(inputs)
        outputs = outputs*255
    print(prof)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))      
    #img = inputs[0,:,:,:].to('cpu')
    #plt.imshow(img.permute(1,2,0))
    #plt.show()
    img = outputs[0,:,:,:].to('cpu').detach().numpy().astype(np.uint8)
    img = img.transpose(1,2,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plt.show()
    #img = targets[0,:,:,:].to('cpu')
    #plt.imshow(img.permute(1,2,0))
    #plt.show()