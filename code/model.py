from typing import DefaultDict
import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss


import os
import random

from utils import Config

# definition of capsule network
class BaseCapNet(nn.Module):
    def __init__(self,K,dynamic):
        super(BaseCapNet, self).__init__()
        self.K = K
        self.item_emb = nn.Embedding(item_num+1, hiddim, padding_idx=0)
        self.user_emb = nn.Embedding(user_num + 1, hiddim, padding_idx=0)
        self.create_mask = {}
        #capsule
        # self.transes = [nn.Linear(hiddim * 2, hiddim) for _ in range(max_K)]
        self.trans = nn.Linear(hiddim * 2, hiddim)
        # self.transes = [nn.Linear(hiddim, hiddim) for _ in range(max_K)]
        # self.trans = nn.Linear(hiddim, hiddim)

        self.dnn = nn.Sequential(
            nn.Linear(hiddim*2,hiddim),
            nn.ReLU(),
            nn.Linear(hiddim,1),
            nn.Sigmoid()
        )
        


    def forward(self, user_id, hist, tgt, histlen, tgtlen, caps, Ks, oldKs, proj, hist_mask, tgt_mask, memflag):
        if memflag:
            hist = torch.cat([hist, tgt], dim=1)
            hist_mask = torch.cat([hist_mask, tgt_mask], dim=1)
        else:
            neg = []
            for i in range(user_id.shape[0]):
                neg_list = random.choices(neg_set[user_id[i]], k=int(nsr * tgtlen[i]))
                neg.append(torch.nn.functional.pad(torch.tensor(neg_list, dtype=torch.int64), (0, tgt.shape[1] - tgtlen[i]), 'constant', 0))
            neg = torch.stack(neg).to(device)
            neg_mask = (neg != 0)

        mbk = max(Ks)  #max K in batch
        capsules = []
        cap_mask = []
        old_cap_mask = []
        for i in range(user_id.shape[0]):
            capsules.append(torch.stack(caps[i] + [torch.zeros([hiddim], dtype=torch.float32).to(device)] * (mbk - Ks[i])))
            cap_mask.append([True]*Ks[i]+[False]*(mbk-Ks[i]))
            old_cap_mask.append([True]*oldKs[i]+[False]*(mbk-oldKs[i]))
        capsules = torch.stack(capsules).to(device).unsqueeze(3)
        cap_mask = torch.tensor(cap_mask).to(device).unsqueeze(2)
        old_cap_mask = torch.tensor(old_cap_mask).to(device).unsqueeze(2)
        

        hist_emb = self.item_emb(hist)
        tgt_emb = self.item_emb(tgt)
        if not memflag:
            neg_emb = self.item_emb(neg)
        user_emb = self.user_emb(user_id)
        #tgt_emb:batchsize*maxlen*hiddim
        # hist_hat = torch.stack([trans(torch.cat([hist_emb,user_emb.repeat([1,hist_emb.shape[1],1])],dim=2)) for trans in self.transes[:mbk]]).permute([1,0,2,3])
        hist_hat = self.trans(torch.cat([hist_emb,user_emb.repeat([1,hist_emb.shape[1],1])],dim=2)).unsqueeze(1).repeat([1,mbk,1,1])
        # hist_hat = self.trans(hist_emb).unsqueeze(1).repeat([1,mbk,1,1])
        # hist_hat = torch.stack([trans(hist_emb) for trans in self.transes[:mbk]]).permute([1,0,2,3])
        hist_hat_iter = hist_hat.detach()        
        #hist_hat: batchsize*K*maxlen*hiddim
        #capsules: batchsize*K*hiddim*1
        
        a = torch.matmul(hist_hat_iter, capsules).squeeze()
        #a: batchsize*K*maxlen
        for iter in range(3):
            b = torch.softmax(a.masked_fill_(~cap_mask, -float('inf')), dim=1)  #force one item into one interest but not one interest is determined by just one or a few items
            #b: batchsize*K*maxlen
            if iter<2:                
                c = torch.matmul(b.unsqueeze(2), hist_hat_iter)#zero item has zero embedding and is no contribute to c, so no padding needed
                #c:batchsize*K*1*hiddim
                if proj:
                    temp1 = torch.matmul(c.squeeze(), c.squeeze().permute([0,2,1]))
                    temp2 = torch.inverse(temp1 + 4e-3*torch.eye(mbk).to(device))
                    caps_proj = torch.matmul(torch.matmul(c.squeeze().permute([0,2,1]),temp2),c.squeeze())
                    c = c - torch.matmul(c.squeeze(),caps_proj).unsqueeze(2)*(~old_cap_mask).unsqueeze(3)
                capsules = (torch.norm(c, dim=3,keepdim=True) / (torch.norm(c, dim=3,keepdim=True)** 2 + 1) * c).permute([0, 1, 3, 2])
                #capsules: batchsize*K*hiddim*1
                
                
                a = torch.matmul(hist_hat_iter, capsules).squeeze() + a
                #a: batchsize*K*maxlen
            else:
                c = torch.matmul(b.unsqueeze(2), hist_hat)#zero item has zero embedding and is no contribute to c, so no padding needed
                #c:batchsize*K*1*hiddim
                if proj:
                    temp1 = torch.matmul(c.squeeze(), c.squeeze().permute([0,2,1]))
                    temp2 = torch.inverse(temp1 + 2e-2*torch.eye(mbk).to(device))
                    caps_proj = torch.matmul(torch.matmul(c.squeeze().permute([0,2,1]),temp2),c.squeeze())
                    c = c - torch.matmul(c.squeeze(),caps_proj).unsqueeze(2)*(~old_cap_mask).unsqueeze(3)
                
                capsules = (torch.norm(c, dim=3,keepdim=True) / (torch.norm(c, dim=3,keepdim=True)** 2 + 1) * c).permute([0, 1, 3, 2])
                #capsules: batchsize*K*hiddim*1
            
        if memflag:
            # created_mask = (torch.max(b, dim=1).values > create_trd) * torch.cat([hist_mask, tgt_mask], dim=1)
            #calculate KL
            create_score = (torch.log(torch.sum(b, dim=1)) - torch.mean(torch.log(b), dim=1)).masked_fill_(~hist_mask,float('inf'))
            create = torch.sum(create_score<create_trd,dim=1)>create_trd2
            # create = torch.sum(torch.max(b, dim=1).values < create_trd, dim=1) > create_trd2+b.shape[2]-histlen-tgtlen
            return capsules.squeeze()*cap_mask, create
        attn = torch.softmax(torch.matmul(tgt_emb.unsqueeze(1), capsules).squeeze().masked_fill_(~cap_mask, -float('inf')),dim=1).permute([0,2,1])
        #attn:batchsize*maxlen*K
        attn_capsules = torch.matmul(attn, capsules.squeeze())
        #attn_capsules:batchsize*maxlen*hiddim
        loss_pos = torch.log(torch.sigmoid(torch.sum(tgt_emb*attn_capsules,dim=2)))*tgt_mask
        loss_neg = torch.log(1-torch.sigmoid(torch.sum(neg_emb*attn_capsules,dim=2)))*neg_mask

        #dnn
        # loss_pos = torch.log(self.dnn(torch.cat([tgt_emb,user_emb.repeat([1,tgt_emb.shape[1],1])],dim=2))).squeeze()*tgt_mask
        # loss_neg = torch.log(1-self.dnn(torch.cat([neg_emb,user_emb.repeat([1,neg_emb.shape[1],1])],dim=2))).squeeze()*neg_mask
        return loss_pos,loss_neg

    def testing(self, user_id, hist, tgt, histlen, tgtlen, caps, Ks, oldKs, proj, hist_mask, tgt_mask):
        neg = []
        for i in range(user_id.shape[0]):
            neg_list = random.choices(neg_set[user_id[i]], k=int(nsr * tgtlen[i]))
            neg.append(torch.nn.functional.pad(torch.tensor(neg_list, dtype=torch.int64), (0, tgt.shape[1] - tgtlen[i]), 'constant', 0))
        neg = torch.stack(neg).to(device)
        neg_mask = (neg != 0)

        mbk = max(Ks)  #max K in batch
        capsules = []
        cap_mask = []
        for i in range(user_id.shape[0]):
            capsules.append(torch.stack(caps[i] + [torch.zeros([hiddim], dtype=torch.float32).to(device)] * (mbk - Ks[i])))
            cap_mask.append([True]*Ks[i]+[False]*(mbk-Ks[i]))
        capsules = torch.stack(capsules).to(device).unsqueeze(3)
        cap_mask = torch.tensor(cap_mask).to(device).unsqueeze(2)
        tgt_emb = self.item_emb(tgt)
        neg_emb = self.item_emb(neg)
        attn = torch.softmax(torch.matmul(tgt_emb.unsqueeze(1), capsules).squeeze().masked_fill_(~cap_mask, -float('inf')),dim=1).permute([0,2,1])
        #attn:batchsize*maxlen*K
        attn_capsules = torch.matmul(attn, capsules.squeeze())
        #attn_capsules:batchsize*maxlen*hiddim
        loss_pos = torch.log(torch.sigmoid(torch.sum(tgt_emb*attn_capsules,dim=2)))*tgt_mask
        loss_neg = torch.log(1-torch.sigmoid(torch.sum(neg_emb*attn_capsules,dim=2)))*neg_mask
        return loss_pos,loss_neg

        
