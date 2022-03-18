import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18


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

        self.visual_net = resnet18(pretrained=True)


        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.fc = nn.Linear(1024, 512)
        self.fc_aq = nn.Linear(512, 512)
        self.fc_vq = nn.Linear(512, 512)

        self.linear11 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(512, 512)

        self.linear21 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(512, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(512)

        self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)

        # question
        self.question_encoder = QstEncoder(93, 512, 512, 1, 512)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(512, 42)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl=nn.Linear(1024,512)


        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()


    def forward(self, audio,visual_posi,visual_nega, question):    
        # print("net audio input: ", audio.shape)
        # print("net question input: ", question.shape)
        ## question features
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        ## audio features  B T,128
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)  
        audio_feat_flag = audio_feat

        ## visua: [2*B*T, 512,14,14]
        # print("v feat1: ", visual_posi.shape)   # [64, 10, 512, 14, 14]
        # v_feat = self.avgpool(visual_posi)
        # print("v feat: ", v_feat.shape)
        # posi_visual_feat=v_feat.squeeze(-1).squeeze(-1) # B T 512


        B,T,C,H,W=visual_posi.size()
        temp_visual=visual_posi.view(B*T,C,H,W)
        v_feat=self.avgpool(temp_visual)
        # print("v_feat: ", v_feat.shape) # [640, 512, 1, 1]
        posi_visual_feat=v_feat.squeeze(-1).squeeze(-1) # B T 512
        posi_visual_feat=posi_visual_feat.view(audio_feat.size(0),-1,C)
        # print("posi_visual_feat: ", posi_visual_feat.shape) # [64, 10, 512]

        # T,C,H,W=visual_posi.size()
        # visual_nega=torch.zeros(T,C,H,W)


        out_match = None
        match_label=None


        # print("posi_visual_feat: ", posi_visual_feat.shape)
        visual_feat_grd=posi_visual_feat.permute(1,0,2)
        
        ## attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)
    
        # attention, question as query on audio
        audio_feat = audio_feat.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None,key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)
        
        feat = torch.cat((audio_feat_att, visual_feat_att), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        ## fusion with question
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)    # [batch_size, ans_vocab_size]

        return out_qa,out_match,match_label
