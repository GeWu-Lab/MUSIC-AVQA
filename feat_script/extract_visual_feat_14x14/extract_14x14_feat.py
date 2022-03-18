import sys 
sys.path.append("/home/guangyao_li/projects/avqa/feat_script/") 
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
from extract_visual_feat.nets_14x14_feat import AVQA_Fusion_Net
from torchvision import transforms


print("\n\n---------------------- extract_14x14_feats -------------------\n\n")

C, H, W = 3, 224, 224


def TransformImage(img):

    transform_list = []
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    transform_list.append(transforms.Resize([224,224]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    trans = transforms.Compose(transform_list)
    frame_tensor = trans(img)
    
    return frame_tensor


def load_frame_info(img_path):

    img = Image.open(img_path).convert('RGB')
    frame_tensor = TransformImage(img)

    return frame_tensor

def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()
    dir_fc = os.path.join(os.getcwd(), params['output_dir'])
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)

    video_list = os.listdir(params['video_path'])    
    list_len = len(video_list)
    print(list_len)

    cnt = 0
    for video in video_list:

        print("name: ", video)
        outfile1 = os.path.join(dir_fc, video + '.npy')
        if os.path.exists(outfile1):
            print(video, " is already exist!")
            continue

        dst = video

        ### image
        select_img = []
        image_list = sorted(glob.glob(os.path.join(params['video_path'], dst, '*.jpg')))
        # samples = np.round(np.linspace(0, len(image_list) - 1, params['n_frame_steps']))
        print("img len: ", len(image_list))
        samples = np.round(np.linspace(0, len(image_list) - 1, len(image_list)))
        
        image_list = [image_list[int(sample)] for sample in samples]
        image_list=image_list[::18]
        for img in image_list:
            frame_tensor_info = load_frame_info(img)
            select_img.append(frame_tensor_info.cpu().numpy())
        select_img=np.array(select_img)
        select_img=torch.from_numpy(select_img)

        select_img=select_img.unsqueeze(0)


        with torch.no_grad():
            visual_out = model(select_img.cuda())
        fea = visual_out.cpu().numpy()

        # print('fea shape', fea.shape)
        # Save the inception features
        outfile_before = os.path.join(dir_fc, video + '.npy')
        np.save(outfile_before, fea)
        
        cnt = cnt + 1
        print("----------------->> ", cnt, " / ", list_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0, 1',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str, default='/home/guangyao_li/dataset/TVQA/frames/frames_hq/rs18/bbt_rs18_14x14', 
                        help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=480,
                        help='how many frames to sampler per video')
    parser.add_argument("--video_path", dest='video_path', type=str, default='/home/guangyao_li/dataset/TVQA/frames/frames_hq/frames/bbt_frames', 
                        help='path to video dataset')
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
        model = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'nasnetalarge':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))

    model=AVQA_Fusion_Net()
    model = nn.DataParallel(model)
    model = model.cuda()
    
    extract_feats(params, model, load_image_fn)
