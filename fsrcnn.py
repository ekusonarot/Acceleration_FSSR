import argparse
import cv2
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import warnings
from pathlib import Path
from models import FSRCNN
from utils import Converter



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir", help="Set input-images dir", type=str, default="./LR")
    parser.add_argument("-o","--output_dir", help="Set output-images dir", type=str, default="./SR")
    parser.add_argument("-s","--scale", help="Select upscale rate", type=int, default=4, choices=range(2, 5))
    args = parser.parse_args()

    indir = args.input_dir
    outdir = args.output_dir
    weight = "./weights/x" + str(args.scale) + "_fsrcnn.pth"
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(outdir, exist_ok="True")

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN(scale_factor=args.scale).to(device)
    state_dict = model.state_dict()

    for n, p in torch.load(weight, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    filenames = [str(filename) for filename in Path(indir).glob('*') if filename.suffix in ['.bmp', '.jpg', '.png']]
    print('\n--- FSRCNN start ---\n')
    for filename in filenames:
        image = cv2.imread(filename).astype(np.float32)
        bicubic = cv2.resize(image, None, fx = args.scale, fy = args.scale, interpolation = cv2.INTER_CUBIC)

        Luminance = Converter.convert_bgr_to_y(image)
        Luminance = torch.from_numpy(Luminance).to(device)
        Luminance = Luminance.unsqueeze(0).unsqueeze(0)

        ycbcr = Converter.convert_bgr_to_ycbcr(bicubic)

        with torch.no_grad():
            preds = model(Luminance).mul(255.0)

        preds = preds.cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(Converter.convert_ycbcr_to_bgr(output), 0.0, 255.0).astype(np.uint8)

        basename = os.path.basename(filename)
        cv2.imwrite(outdir + "/FSRCNN_x" + str(args.scale) + basename, output)
        print("output {}".format("FSRCNN_x" + str(args.scale) + basename))
    print("\n[ {} SR pictures are output ]".format(len(filenames)))
