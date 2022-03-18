import shutil
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
import pretrainedmodels
from pretrainedmodels import utils

C, H, W = 3, 224, 224

def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()
    dir_fc = os.path.join(os.getcwd(), params['output_dir'])
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)

    video_list = os.listdir(params['video_path'])
    nn = 0
    total_len = len(video_list)
    for video in video_list:

        nn = nn + 1
        dst = video
        print("\n-->: ", video)

        if video == 'yz02dWv_shs':
            print(video, " is too large!")
            continue

        outfile = os.path.join(dir_fc, video + '.npy')
        if os.path.exists(outfile):
            print(video, " is already processed!")
            continue

        image_list = sorted(glob.glob(os.path.join(params['video_path'], dst, '*.jpg')))
        # print("n_frame_steps: ", params['n_frame_steps'])
        # samples = np.round(np.linspace(0, len(image_list) - 1, params['n_frame_steps']))

        params_frames = len(image_list)
        # print("params_frames: ", params_frames)
        samples = np.round(np.linspace(0, len(image_list) - 1, params_frames))

        image_list = [image_list[int(sample)] for sample in samples]
        # print("image_list len: ", len(image_list))
        images = torch.zeros((len(image_list), C, H, W))
        i = 0
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img


        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # print(img_feats.shape)
        # Save the inception features
        # outfile = os.path.join(dir_fc, video + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        #shutil.rmtree(dst)
        print("Process: ", nn, " / ", total_len, " ------- video id: ", video, " ------- save shape: ", img_feats.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='/home/guangyao_li/dataset/LFAV_dataset/feat/visual-feat-res18', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=80,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='/home/guangyao_li/dataset/LFAV_dataset/update/update_frames/', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet18',
                        help='the CNN model you want to use to extract_feats')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'resnet18':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet18(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'vgg19_bn':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.vgg19_bn(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'nasnetalarge':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)

    model = model.cuda()
    extract_feats(params, model, load_image_fn)
