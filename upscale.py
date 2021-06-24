import argparse
import cv2
import os
import warnings
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir", help="Set input-images dir", type=str, default="./LR")
    parser.add_argument("-o","--output_dir", help="Set output-images dir", type=str, default="./UP")
    parser.add_argument("-s","--scale", help="Select upscale rate", type=int, default=4, choices=range(2, 5))
    parser.add_argument("-m","--method",  help="Choice upscale method", type=int, default=2, choices=range(0, 4))
    args = parser.parse_args()

    indir = args.input_dir
    outdir = args.output_dir
    method = args.method + 1 if args.method == 3 else args.method
    algorithm = ['NEA','LIN','CUB','','LAN']
    os.makedirs(outdir, exist_ok="True")
    filenames = [str(filename) for filename in Path(indir).glob('*') if filename.suffix in ['.bmp', '.jpg', '.png']]
    print('\n--- UPSCALE start ---\n')
    for filename in filenames:
        image = cv2.imread(filename)
        output = cv2.resize(image, None, fx = args.scale, fy = args.scale, interpolation = cv2.INTER_CUBIC)
        basename = os.path.basename(filename)
        cv2.imwrite(outdir + os.sep + algorithm[method] + '_x' + str(args.scale) + basename, output)
        print("output {}".format(algorithm[method] + '_x' + str(args.scale) + basename))
    print("\n[ {} Upscale pictures are output ]".format(len(filenames)))