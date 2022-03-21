from __future__ import print_function
import sys 
sys.path.append("/home/guangyao_li/projects/music_avqa/") 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from rebuttal_avqtg_qa_grd_finetinue_baseline_v2.dataloader_qa_grd_baseline import *
from rebuttal_avqtg_qa_grd_finetinue_baseline_v2.nets_qa_grd_baseline import AVQA_Fusion_Net
import ast
import json
import numpy as np
import pdb

import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/rebuttal_avqtg_qa_grd_finetinue_baseline_v2/'+TIMESTAMP)

print("\n--------------- main rebuttal AV+Q+TG rebuttal_avqtg_qa_grd_finetinue_baseline_v2 --------------- \n")

def batch_organize(audio_data, posi_img_data, nega_img_data):

    # print("audio data: ", audio_data.shape)
    (B, T, C) = audio_data.size()
    audio_data_batch=audio_data.view(B*T,C)
    batch_audio_data = torch.zeros(audio_data_batch.shape[0] * 2, audio_data_batch.shape[1])


    (B, T, C, H, W) = posi_img_data.size()
    posi_img_data_batch=posi_img_data.view(B*T,C,H,W)
    nega_img_data_batch=nega_img_data.view(B*T,C,H,W)


    batch_image_data = torch.zeros(posi_img_data_batch.shape[0] * 2, posi_img_data_batch.shape[1], posi_img_data_batch.shape[2],posi_img_data_batch.shape[3])
    batch_labels = torch.zeros(audio_data_batch.shape[0] * 2)
    for i in range(audio_data_batch.shape[0]):
        batch_audio_data[i * 2, :] = audio_data_batch[i, :]
        batch_audio_data[i * 2 + 1, :] = audio_data_batch[i, :]
        batch_image_data[i * 2, :] = posi_img_data_batch[i, :]
        batch_image_data[i * 2 + 1, :] = nega_img_data_batch[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return batch_audio_data, batch_image_data, batch_labels




def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

        # audio_batch,visual_batch,match_label=batch_organize(audio,visual_posi, visual_nega)

        # audio_batch,visual_batch,match_label = audio_batch.type(torch.FloatTensor).cuda(), \
        #                                     visual_batch.type(torch.FloatTensor).cuda(), \
        #                                     match_label.type(torch.LongTensor).cuda()

        # print("audio batch: ", audio_batch.shape)
        # print("visual_batch: ", visual_batch.shape)
        optimizer.zero_grad()
        out_qa, out_match,match_label = model(audio, visual_posi,visual_nega, question)
        # print("out_qa shape: ", out_qa.shape)
        # print("out_match shape: ", out_match.shape)

        # output.clamp_(min=1e-7, max=1 - 1e-7)
        loss_qa = criterion(out_qa, target)

        loss=loss_qa

        writer.add_scalar('data/both',loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader,epoch):
    model.eval()
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

            # audio_batch,visual_batch,match_label=batch_organize(audio,visual_posi, visual_nega)
            # audio_batch,visual_batch,match_label = audio_batch.type(torch.FloatTensor), \
            #                                 visual_batch.type(torch.FloatTensor), \
            #                                 match_label.type(torch.LongTensor)

            preds_qa,preds_match,match_label = model(audio, visual_posi,visual_nega, question)

            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()


    print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))


    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa


def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('../dataset/avqa-test.json', 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []

    que_id=[]
    pred_results=[]
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

            # audio_batch,visual_batch,match_label=batch_organize(audio,visual_posi, visual_nega)
            # audio_batch,visual_batch,match_label = audio_batch.type(torch.FloatTensor), \
            #                                 visual_batch.type(torch.FloatTensor), \
            #                                 match_label.type(torch.LongTensor)

            preds_qa,preds_match,match_label = model(audio, visual_posi,visual_nega, question)
            # preds = model(audio, video, question)
            preds=preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            # save pred results
            pred_bool=predicted == target
            for index in range(len(pred_bool)):
                pred_results.append(pred_bool[index].cpu().item())
                que_id.append(question_id[index].item())

            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                    # AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())

    with open("pred_results/AVQA_AVatt_Net.txt", 'w') as f:
        print("len q: ", len(que_id))
        print("len pred: ", len(pred_results))
        for index in range(len(que_id)):
            # print("que_id: ", str(que_id[index]))
            # print("pred: ", str(pred_bool[index]))
            f.write(str(que_id[index])+' '+str(pred_results[index]) + '\n')

    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default='/home/guangyao_li/dataset/avqa-features/feats/vggish', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-8fps', help="video dir")
    parser.add_argument(
        "--gt_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-1fps-bbox-p0.5/bbox_frame_count_gt', help="gt dir")
    parser.add_argument(
        "--st_dir", type=str, default='/home/guangyao_li/dataset/avqa-features/feats/r2plus1d_18', help="video dir")
    

    parser.add_argument(
        "--label_train", type=str, default="../dataset/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="../dataset/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="../dataset/avqa-test.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='rebuttal_avqtg_qa_grd_finetinue_baseline_v2/models_av_q_tg/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='rebuttal_av_q_tg', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')


    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net()
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(gt_dir=args.gt_dir, label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataset = AVQA_dataset(gt_dir=args.gt_dir, label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


        # ===================================== load pretrained model ===============================================

        # ####### concat model
        # pretrained_file = "/home/guangyao_li/projects/avqa/avqa-code-cvpr2022-0922/grounding_localization/grounding_gen/models_grounding_gen/main_grounding_gen_best.pt"
        # checkpoint = torch.load(pretrained_file)
        # print("\n-------------- loading pretrained models --------------")
        # # print("load before: ", checkpoint['module.fc_a1.weight'])
        # model_dict = model.state_dict()
        # tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
        # pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
        # # print("\n", len(pretrained_dict1))

        # model_dict.update(pretrained_dict1) #利用预训练模型的参数，更新模型
        # model.load_state_dict(model_dict)
        # # print("load after: ", model.module.fc_a1.weight)

        # print("\n-------------- load pretrained models --------------")

        # ===================================== load pretrained model ===============================================

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader,epoch)
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")

    else:
        test_dataset = AVQA_dataset(gt_dir=args.gt_dir, label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                   st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        test(model, test_loader)


if __name__ == '__main__':
    main()