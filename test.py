from PIL import Image
from models import FSRCNN
from classification import CategoricalCNN
import torch
from torchvision import transforms
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img = Image.open("image.jpg")
img = transforms.Resize(size=(180, 280))(img)
img = transforms.ToTensor()(img)
img = torch.unsqueeze(img, 0).to(device)
print(img.shape)
fsrcnn = FSRCNN(4, 3).to(device)
light_fsrcnn = FSRCNN(4, 3, d=32, m=1).to(device)
fsrcnn.load_state_dict(torch.load("./weights/x4_3ch_fsrcnn_9.pth", map_location=torch.device(device)))
light_fsrcnn.load_state_dict(torch.load("./weights/x4_3ch_lightfsrcnn_9.pth", map_location=torch.device(device)))

classify = CategoricalCNN(light_fsrcnn, fsrcnn, 4, input_shape=(img.size(1), img.size(2), img.size(3)), block_size=20, device=device).to(device)
classify.load_state_dict(torch.load("./test.pth", map_location=torch.device(device)), strict=False)
classify.eval()

with torch.no_grad():
    output, _ = classify(img)
    output = output[0,:,:,:].to('cpu').detach().permute(1, 2, 0).numpy().copy()
    print(output.shape)
    cv2.imshow("image", output)
    cv2.waitKey(0)
    cv2.destroyWindow("image")