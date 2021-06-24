import cv2
import os
from utils import LPIPSpreprocess
import lpips
from skimage.metrics import structural_similarity as ssim
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="set Input-image path", type=str, required=True)
    parser.add_argument("-gt","--ground_truth", help="set Ground-Truth Image path", type=str, required=True)
    args = parser.parse_args()

    img_path = args.input
    gt_path = args.ground_truth

    image = cv2.imread(img_path)
    gt = cv2.imread(gt_path)

    lpips_model = lpips.LPIPS(net='alex')

    PSNR = cv2.PSNR(image, gt)
    SSIM = ssim(image, gt, multichannel = True)
    LPIPS = lpips_model(LPIPSpreprocess(image), LPIPSpreprocess(gt))[0,0,0,0].item()

    print("\n ---Similalities between [{}] and [{}] ---\n   PSNR  : {}\n   SSIM  : {}\n   LPIPS : {}"\
.format(os.path.basename(img_path), os.path.basename(gt_path),PSNR,SSIM,LPIPS))