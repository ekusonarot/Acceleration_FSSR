import torch
from torch import nn
from torch.nn.modules import padding
from models import FSRCNN
import math
import threading

padding = 6

class CategoricalCNN(nn.Module):
    def __init__(self, light_model, complex_model, scale_factor=4, input_shape=(3, 512, 288), block_size=20, device='cpu'):
        super(CategoricalCNN, self).__init__()
        self.input_shape = input_shape
        self.block_size = block_size
        self.num_blocks = input_shape[1]*input_shape[2]//block_size**2
        self.scale_factor = scale_factor
        self.light_model = light_model
        self.complex_model = complex_model
        last_dim = input_shape[1]*input_shape[2]//16
        self.device=device
        self.pad = nn.ZeroPad2d(padding=padding)
        self.class_first_part = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=[3//2, 3//2], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=[3//2, 3//2], padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.class_second_part = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=[3//2, 3//2], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=[3//2, 3//2], padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(8, 1, kernel_size=block_size//4, stride=block_size//4),
            nn.Flatten()
        )
        self.class_last_part = nn.Sequential(
            #nn.Linear(last_dim, self.num_blocks),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.class_first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.class_second_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.class_last_part:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
                nn.init.zeros_(m.bias.data)

    def process_lightmodel(self, blocked_img, light_mask):
        global output_lightmodel
        input_lightmodel = torch.matmul(blocked_img.transpose(0,3),light_mask.float()).transpose(0,3)
        output_lightmodel = self.light_model(input_lightmodel).clamp(0.0, 1.0)
        output_lightmodel = torch.matmul(output_lightmodel.transpose(0,3),light_mask.transpose(0,1).float()).transpose(0,3)

    def process_complexmodel(self, blocked_img, complex_mask):
        global output_complexmodel
        input_complexmodel = torch.matmul(blocked_img.transpose(0,3),complex_mask.float()).transpose(0,3)
        output_complexmodel = self.complex_model(input_complexmodel).clamp(0.0, 1.0)
        output_complexmodel = torch.matmul(output_complexmodel.transpose(0,3),complex_mask.transpose(0,1).float()).transpose(0,3)

    def forward(self, input):
        x = self.class_first_part(input)
        x = self.class_second_part(x)
        class_vector = self.class_last_part(x)
        class_vector = class_vector.reshape(-1)
        input = self.pad(input)
    
        blocked_img = torch.reshape(input.unfold(2,self.block_size+2*padding,self.block_size).unfold(3,self.block_size+2*padding,self.block_size).transpose(1,2).transpose(2,3),
            (-1, 
            self.input_shape[0],
            self.block_size+2*padding,
            self.block_size+2*padding))

        mask = torch.where(class_vector>0.5, 1, 0)
        complex_mask = mask.nonzero()
        light_mask = (1-mask).nonzero()

        complex_mask = torch.nn.functional.one_hot(complex_mask, num_classes=blocked_img.size(0)).squeeze(1).transpose(0,1)
        light_mask = torch.nn.functional.one_hot(light_mask, num_classes=blocked_img.size(0)).squeeze(1).transpose(0,1)

        global output_lightmodel, output_complexmodel
        output_lightmodel = torch.zeros(1).to(self.device)
        output_complexmodel = torch.zeros(1).to(self.device)

        if self.device == 'cpu':
            thread1 = threading.Thread(target=self.process_lightmodel, args=(blocked_img, light_mask))
            thread2 = threading.Thread(target=self.process_complexmodel, args=(blocked_img, complex_mask))
            if light_mask.size(-1) != 0:
                thread1.start()
            if complex_mask.size(-1) != 0:
                thread2.start()
            if light_mask.size(-1) != 0:
                thread1.join()
            if complex_mask.size(-1) != 0:
                thread2.join()
        else:
            if light_mask.size(-1) != 0:
                self.process_lightmodel(blocked_img, light_mask)
            if complex_mask.size(-1) != 0:
                self.process_complexmodel(blocked_img, complex_mask)
        output = torch.add(output_lightmodel, output_complexmodel)\
            .narrow(2, self.scale_factor*padding, self.scale_factor*self.block_size)\
            .narrow(3, self.scale_factor*padding, self.scale_factor*self.block_size)
        output = torch.reshape(output.transpose(0,1).transpose(1,2),
            (input.size(1),
            self.block_size*self.scale_factor,
            -1,
            self.input_shape[2]*self.scale_factor))
        output = torch.reshape(output.transpose(1,2),
            (input.size(1),
            -1,
            self.input_shape[1]*self.scale_factor,
            self.input_shape[2]*self.scale_factor)).transpose(0,1)
        return output, class_vector
    
    def Loss(self, outputs, targets, class_vector):
        outputs = torch.clamp(outputs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)
        blocked_outputs = torch.reshape(outputs.unfold(2,self.block_size,self.block_size).unfold(3,self.block_size,self.block_size).transpose(1,2).transpose(2,3),
            (-1, 
            self.input_shape[0],
            self.block_size*self.scale_factor,
            self.block_size*self.scale_factor))
        blocked_targets = torch.reshape(targets.unfold(2,self.block_size,self.block_size).unfold(3,self.block_size,self.block_size).transpose(1,2).transpose(2,3),
            (-1, 
            self.input_shape[0],
            self.block_size*self.scale_factor,
            self.block_size*self.scale_factor))
        loss1 = torch.sub(blocked_outputs, blocked_targets)
        loss1 = torch.pow(loss1,2)
        loss1 = torch.sqrt(loss1)
        loss1 = torch.sum(loss1, dim=(1,2,3))
        loss1 = torch.mul(loss1, class_vector)
        loss1 = torch.sum(loss1, dim=0)
        loss1 = torch.div(loss1,\
            blocked_targets.size(0)*\
            blocked_targets.size(1)*\
            blocked_targets.size(2)*\
            blocked_targets.size(3))*25

        loss2 = torch.sub(1,class_vector)
        loss2 = torch.sum(loss2,dim=0)
        loss2 = torch.div(loss2, class_vector.size(0))
        
        loss = torch.add(loss1, loss2)
        return loss
        

        
if __name__ == '__main__':
    fsrcnn = FSRCNN(scale_factor=4, num_channels=3).to('cpu')
    net = CategoricalCNN(fsrcnn,fsrcnn)
    img = torch.arange(0,7776000.)
    targets = torch.arange(0,124416000)
    img = torch.reshape(img,(5,3,540,960))
    targets = torch.reshape(targets, (5,3,2160,3840))
    outputs, class_vector = net(img)
    loss = net.Loss(outputs,targets,class_vector)
    loss.backward()