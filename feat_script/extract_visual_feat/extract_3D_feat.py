import shutil
import sys
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import transforms as TF
import utils
import torchvision


C, H, W = 3, 112, 112

def extract_feats(params, model, load_img):
    global C, H, W
    model.eval()
    dir_fc = os.path.join(os.getcwd(), params['output_dir'])
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)

    video_list = os.listdir(params['video_path'])
    nn = 0
    total_len = len(video_list)
    for video in video_list:
        print("\n-->: ", video)

        nn = nn + 1
        dst = video

        if video == 'yz02dWv_shs':
            print(video, " is too large!")
            continue

        outfile = os.path.join(dir_fc, video + '.npy')
        if os.path.exists(outfile):
            print(video, " is already processed!")
            continue

        image_list = sorted(glob.glob(os.path.join(params['video_path'], dst, '*.jpg')))
        
        # samples = np.round(np.linspace(0, len(image_list) - 1, params['n_frame_steps']))

        params_frames = len(image_list)
        samples = np.round(np.linspace(0, len(image_list) - 1, params_frames))

        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list)//1, C, 1, H, W))
        i = 0
        for iImg in range(len(image_list)):

            ii = i//1
            img = load_img(image_list[iImg])
            images[ii, :, i%1, :, :] = img
            i += 1

        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        # outfile = os.path.join(dir_fc, video + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        #shutil.rmtree(dst)
        # print(nn)
        print("Process: ", nn, " / ", total_len, " ------- video id: ", video, " ------- save shape: ", img_feats.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='1',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='./data/feats/r2plus1d', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=80,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='./data/frames/', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='r2plus1d_18',
                        help='the CNN model you want to use to extract_feats')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'r2plus1d_18':
        model = models.video.r2plus1d_18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        for param in model.parameters():
            param.requires_grad = False
        T, C, H, W = 1, 3, 112, 112
        load_img = utils.LoadTransformImage()

    else:
        print("doesn't support %s" % (params['model']))

    model = nn.DataParallel(model)
    model = model.cuda()
    extract_feats(params, model, load_img)
