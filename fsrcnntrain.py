from torch.utils.data import DataLoader
import dataset
import models
import torch
from torch import optim
import torchvision.transforms as transforms

lr = 1e-3
batchsize=32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
fsrcnn = models.FSRCNN(4, 3).to(device)
#fsrcnn.load_state_dict(torch.load("./weights/x4_3ch_fsrcnn_9.pth", map_location=torch.device(device)))

optimizer = optim.Adam([
    {'params': fsrcnn.first_part.parameters()},
    {'params': fsrcnn.mid_part.parameters(), 'lr': lr*0.1},
    {'params': fsrcnn.last_part.parameters(), 'lr': lr*0.05}
    ], lr=lr)

transform_input = transforms.Compose([
    transforms.Resize((320,180)),
    transforms.ToTensor()
    ])
transform_target = transforms.Compose([
    transforms.Resize((1280,720)),
    transforms.ToTensor()
])
dataset = dataset.MyDataset("./train_sharp", transform_input, transform_target)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

criterion = torch.nn.MSELoss()

fsrcnn.train()
for i in range(0,10):
    print("epoch: {}".format(i))
    j=0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs= fsrcnn(inputs).clamp(0.0, 1.0)
        loss = criterion(outputs, targets.clamp(0.0, 1.0))
        loss.backward()
        optimizer.step()
        j+=1
        if j%20==0:
            print("epoch{}: {}/{}: loss: {}".format(i,j*batchsize,dataloader.__len__()*batchsize,loss))
        del loss
    torch.save(fsrcnn.state_dict(), './weights/x4_3ch_fsrcnn_{}.pth'.format(i))