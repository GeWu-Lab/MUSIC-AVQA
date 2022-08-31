import sys 
sys.path.append("path to MUSIC-AVQA_CVPR2022") 
import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from extract_visual_feat_14x14.visual_net_14x14 import resnet18

# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class AVQA_Fusion_Net(nn.Module):

    def __init__(self):
        super(AVQA_Fusion_Net, self).__init__()



        # for features
        self.fc_a1 =  nn.Linear(128, 512)
        self.fc_a2=nn.Linear(512,512)

        # visual
        self.visual_net = resnet18(pretrained=True)

        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_gl=nn.Linear(1024,512)
        self.tanh = nn.Tanh()


    def forward(self,visual):       

        ## visual, input: [16, 20, 3, 224, 224]
        (B, T, C, H, W) = visual.size()
        visual = visual.view(B * T, C, H, W)                # [320, 3, 224, 224]
        v_feat_out_res18 = self.visual_net(visual)                    # [320, 512, 14, 14]



        return v_feat_out_res18
