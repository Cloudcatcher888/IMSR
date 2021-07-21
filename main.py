import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss



import os
import random

import utils

print(os.path.abspath('.'))
pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 50)
torch.set_printoptions(edgeitems=10, linewidth=200, precision=4)

memory_needed = 8000
device = torch.device('cpu')
import pynvml
pynvml.nvmlInit()
if torch.cuda.is_available():
    for i in range(4): 
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = float(meminfo.free/1024**2)
        print(free_mem)
        if free_mem > memory_needed:
            device = torch.device(f'cuda:{i}')
            break
# device = torch.device('cpu')
print(f'use {device}')
random.seed(1) 
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1) 
torch.cuda.manual_seed_all(1)

item_cate_file = '/home/wangzhikai/recommendation_learning/dataset/amazon/books_item_cate.txt'
train_file = '/home/wangzhikai/recommendation_learning/dataset/amazon/books_30_train.txt'
item_cate = {}
trainset = []
current_user_group = -1
min_timestamp = 1000000000000
max_timestamp = 0
cate_num = 0

merge_num = 1

#data reading
with open(item_cate_file,'r') as item_cate_f, open(train_file,'r') as train_f:
    for line in item_cate_f.readlines():
        cate = int(line.split(',')[1])
        cate_num = max(cate_num, cate)
        item_cate[int(line.split(',')[0])] = cate
    # item_cate[0] = cate_num + 1
    cate_num += 1
    # item 0 is no item, cate[last] is the unknown cate
    print('cate_num',cate_num)
    for line in train_f.readlines():
        user_id, item_id, timestamp = int(line.split(',')[0]), int(line.split(',')[1]), int(line.split(',')[2])
        min_timestamp = min(timestamp, min_timestamp)
        max_timestamp = max(timestamp, max_timestamp)
        if int(user_id/merge_num) != current_user_group:
            current_user_group += 1
            trainset.append([])
        try:
            trainset[current_user_group].append([item_id, item_cate[item_id], timestamp])
        except:
            item_cate[item_id] = cate_num
            trainset[current_user_group].append([item_id, cate_num, timestamp])

user_num = len(trainset)
item_num = max(item_cate)
print(f'user number is {user_num}')
print(f'item number is {item_num}')
for i in range(item_num):
    if i not in item_cate:
        item_cate[i] = cate_num
# print([len(trainset[i]) for i in range(len(trainset))])

for user in trainset:
    user.sort(key=lambda x: x[2])
# print(trainset[0])


#neg set
neg_set = []
for user in trainset:
    neg_set.append(random.choices(list(set(range(1, item_num + 1)) - set([item[0] for item in user])),k=1000))


#task split
min_timestamp -= 100
max_timestamp += 100
task_num = 6
base_task_ratio = 0.85
timespan = (max_timestamp - min_timestamp) * (1 - base_task_ratio) / task_num
incre_time_start = min_timestamp + base_task_ratio * (max_timestamp - min_timestamp)
timezone = [incre_time_start+i*timespan for i in range(task_num+1)]

trainset_split = []
for user in trainset:
    user_split = [[] for _ in range(task_num+1)]
    idx = 0
    for item in user:
        if item[2] > timezone[idx]:
            idx += 1
        user_split[idx].append(item)
    trainset_split.append(user_split)

maxbaselen = 0
maxtasklen = 0
for i, user in enumerate(trainset_split):
    if i < 0:        
        print(i, [len(s) for s in user])
        print(user)
    maxbaselen = max(maxbaselen, len(user[0]))
    maxtasklen = max(maxtasklen, max([len(s) for s in user[1:]]))
print('maxbaselen', maxbaselen)
print('maxtasklen', maxtasklen)


"""
trainset_split:
user1:[
    basetask:[
        item(id,ct,ts), item, item, item, item ,item, item, ...
    ]
    task1:[
        item, item, item...
    ],
    task2:[
        item, item, item...
    ]
    ...
    task6:[
        item, item, item...
    ]
]
user2:[
    ...
]
"""

#hyper parameters
batchsize = 512
K = 4
max_K = cate_num
delta_K = 3
hiddim = 16
hist_ratio = 0.7  # can be larger than 0.7 int is cut

create_trd = 2.5
create_trd2 = 5
proj_btn = True
prune_btn = True
memory = True 



"""
one data in traintask:
userid, histseq, tgtseq
"""

class Traintask(Dataset):
    def __init__(self, trainset_split, task_id):
        self.traintask = [user[task_id] for user in trainset_split]
        if task_id == 0:
            print('set up basetask!')
            self.maxhistlen = int(maxbaselen * hist_ratio) + 1
            self.maxtgtlen = int(maxbaselen * (1 - hist_ratio)) + 1
            self.maxbaselen = self.maxhistlen + self.maxtgtlen
        else:
            print(f'set up task {task_id}')
            self.maxhistlen = int(maxtasklen * hist_ratio) + 1
            self.maxtgtlen = int(maxtasklen * (1 - hist_ratio)) + 1
            self.maxtasklen = self.maxhistlen + self.maxtgtlen
        self.task_id = task_id

    def __getitem__(self, idx):
        userid = idx
        if len(self.traintask[idx]) < 2:
            histseq = random.choices(list(item_cate),k=3)
            tgtseq = random.choices(list(item_cate),k=3)
        else:
            hs = int(len(self.traintask[idx])*hist_ratio)
            histseq = self.traintask[idx][:hs]
            histseq = [r[0] for r in histseq]
            tgtseq = self.traintask[idx][hs:]
            tgtseq = [r[0] for r in tgtseq]
        histlen = len(histseq)
        tgtlen = len(tgtseq)

        #to tensor and padding
        userid_tensor = torch.tensor([userid], dtype=torch.int64)
        histseq_tensor = torch.tensor(histseq, dtype=torch.int64)
        histseq_tensor = torch.nn.functional.pad(histseq_tensor, (0, self.maxhistlen - histlen), 'constant', 0)
        histseq_mask = (histseq_tensor != 0)
        tgtseq_tensor = torch.tensor(tgtseq, dtype=torch.int64)
        tgtseq_tensor = torch.nn.functional.pad(tgtseq_tensor, (0, self.maxtgtlen - tgtlen), 'constant', 0)
        tgtseq_mask = (tgtseq_tensor != 0)
        histlen_tensor = torch.tensor(histlen, dtype=torch.int64)
        tgtlen_tensor = torch.tensor(tgtlen, dtype=torch.int64)

        return userid_tensor, histseq_tensor, tgtseq_tensor, histlen_tensor, tgtlen_tensor, histseq_mask, tgtseq_mask
        
    def __len__(self):
        return len(self.traintask)


nsr = 1/1 # neg sample ratio

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


#memory
memory_pool = [[torch.randn([hiddim], dtype=torch.float32).to(device).detach() for _ in range(K)] for _ in range(user_num)]
memory_len = [len(user) for user in memory_pool]
old_memoryLens = [len(user) for user in memory_pool]

"""
memory pool:
user1: mem1, mem2, (mem3,...)
user2: mem1, mem2, (mem3,...)
"""


#base training
basecapnet = BaseCapNet(K, False).to(device)
opt = torch.optim.Adam(
    basecapnet.parameters(),
    weight_decay = 10e-6,
    lr=10e-3
)

basedataset = Traintask(trainset_split, 0)
baseloader = DataLoader(basedataset, shuffle = True, batch_size = batchsize)

for idx in range(1):
    basecapnet.train()
    
    for epoch in range(20):
        losses = []
        for i, (userid, histseq, tgtseq, histlen, tgtlen, histseq_mask, tgtseq_mask) in enumerate(baseloader):
            Ks = [memory_len[i] for i in userid.flatten().numpy().tolist()]
            capsules = [memory_pool[i] for i in userid.flatten().numpy().tolist()]
            pos_loss, neg_loss = basecapnet(
                    user_id=userid.to(device),
                    hist=histseq.to(device),
                    tgt=tgtseq.to(device),
                    histlen=histlen.to(device),
                    tgtlen=tgtlen.to(device),
                    caps=capsules,
                    Ks=Ks,
                    oldKs = Ks,
                    proj=False,
                    hist_mask=histseq_mask.to(device),
                    tgt_mask=tgtseq_mask.to(device),
                    memflag=False
            )
            loss = -(torch.sum(pos_loss) + torch.sum(neg_loss))/(1+nsr)/(torch.sum(tgtlen))
            opt.zero_grad()
            loss.backward()
            opt.step()
            # if i % 1 == 0:
            #     print(loss.item())
            losses.append(loss.item())
            # break
        print('loss', sum(losses) / len(losses))
    
    with torch.no_grad():
        basecapnet.eval()
        if memory ==True:
            #memory
            memory_loader = DataLoader(Traintask(trainset_split,0),shuffle=False,batch_size=batchsize)
            for i, (userid,histseq,tgtseq,histlen,tgtlen,histseq_mask,tgtseq_mask) in enumerate(memory_loader):
                Ks = [memory_len[i] for i in userid.flatten().numpy().tolist()]
                capsules = [memory_pool[i] for i in userid.flatten().numpy().tolist()]
                caps,create = basecapnet(
                            user_id=userid.to(device),
                            hist=histseq.to(device),
                            tgt=tgtseq.to(device),
                            histlen=histlen.to(device),
                            tgtlen=tgtlen.to(device),
                            caps=capsules,
                            Ks=Ks,
                            oldKs = Ks,
                            proj=False,
                            hist_mask=histseq_mask.to(device),
                            tgt_mask=tgtseq_mask.to(device),
                            memflag=True
                )
                # print(create)
                for j,user in enumerate(userid):
                    user = user.item()
                    for K_id in range(Ks[j]):
                        memory_pool[user][K_id] = caps[j, K_id].detach()

        #evaluation
        #testing on the subsequent items based on the memoried capsules
        total_test_loss = []
        total_test_len = []
        testdataloader = DataLoader(Traintask(trainset_split,1),shuffle=False,batch_size=batchsize)
        for i, (userid,histseq,tgtseq,histlen,tgtlen,histseq_mask,tgtseq_mask) in enumerate(testdataloader):
            Ks = [memory_len[i] for i in userid.flatten().numpy().tolist()]
            capsules = [memory_pool[i] for i in userid.flatten().numpy().tolist()]
            pos_loss, neg_loss = basecapnet.testing(
                    user_id=userid.to(device),
                    hist=histseq.to(device),
                    tgt=histseq.to(device),
                    histlen=histlen.to(device),
                    tgtlen=histlen.to(device),
                    caps=capsules,
                    Ks=Ks,
                    oldKs = Ks,
                    proj=False,
                    hist_mask=histseq_mask.to(device),
                    tgt_mask=histseq_mask.to(device)
            )
            loss = -(torch.sum(pos_loss) + torch.sum(neg_loss))
            loss_len = (1+nsr)*(torch.sum(histlen))
            total_test_loss.append(loss.item())
            total_test_len.append(loss_len.item())
        print(sum(total_test_loss)/sum(total_test_len))
                



for idx in range(1, task_num):
    #new interests detector:
    with torch.no_grad():
        basecapnet.eval()
        memory_loader = DataLoader(Traintask(trainset_split,idx),shuffle=False,batch_size=batchsize)
        for i, (userid,histseq,tgtseq,histlen,tgtlen,histseq_mask,tgtseq_mask) in enumerate(memory_loader):
            Ks = [memory_len[i] for i in userid.flatten().numpy().tolist()]
            capsules = [memory_pool[i] for i in userid.flatten().numpy().tolist()]
            caps, create = basecapnet(
                        user_id=userid.to(device),
                        hist=histseq.to(device),
                        tgt=tgtseq.to(device),
                        histlen=histlen.to(device),
                        tgtlen=tgtlen.to(device),
                        caps=capsules,
                        Ks=Ks,
                        oldKs=Ks,
                        proj=False,
                        hist_mask=histseq_mask.to(device),
                        tgt_mask=tgtseq_mask.to(device),
                        memflag=True
            )
            # print(create)
            # print(Ks)
            for j,user in enumerate(userid):
                user = user.item()
                old_memoryLens[user] = memory_len[user]
                if create[j].item():
                    for iter in range(delta_K):
                        memory_pool[user].append(torch.randn([hiddim], dtype=torch.float32).to(device).detach())
                        memory_len[user] += 1

    taskloader = DataLoader(Traintask(trainset_split, idx), shuffle=True, batch_size=batchsize)

    for epoch in range(10):
        losses = []
        for i, (userid, histseq, tgtseq, histlen, tgtlen, histseq_mask, tgtseq_mask) in enumerate(taskloader):
            # print(i)
            # print(userid) #not user_id  which is a global var
            # print(histseq)
            # print(tgtseq)
            # print(histlen)
            # print(tgtlen)
            # print(histseq_mask)
            # print(tgtseq_mask)
            Ks = [memory_len[i] for i in userid.flatten().numpy().tolist()]
            old_Ks = [old_memoryLens[i] for i in userid.flatten().numpy().tolist()]
            # print(old_Ks)
            capsules = [memory_pool[i] for i in userid.flatten().numpy().tolist()]
            pos_loss, neg_loss = basecapnet(
                    user_id=userid.to(device),
                    hist=histseq.to(device),
                    tgt=tgtseq.to(device),
                    histlen=histlen.to(device),
                    tgtlen=tgtlen.to(device),
                    caps=capsules,
                    Ks=Ks,
                    oldKs = old_Ks,
                    proj=proj_btn,
                    hist_mask=histseq_mask.to(device),
                    tgt_mask=tgtseq_mask.to(device),
                    memflag=False
            )
            loss = -(torch.sum(pos_loss) + torch.sum(neg_loss))/(1+nsr)/(torch.sum(tgtlen))
            opt.zero_grad()
            loss.backward()
            opt.step()
            # if i % 1 == 0:
            #     print(loss.item())
            losses.append(loss.item())
            # break
        print('loss', sum(losses) / len(losses))

    #discard interest
    with torch.no_grad():
        basecapnet.eval()
        if memory==True:
            memory_loader = DataLoader(Traintask(trainset_split,idx),shuffle=False,batch_size=batchsize)
            for i, (userid,histseq,tgtseq,histlen,tgtlen,histseq_mask,tgtseq_mask) in enumerate(memory_loader):
                Ks = [memory_len[i] for i in userid.flatten().numpy().tolist()]
                old_Ks = [old_memoryLens[i] for i in userid.flatten().numpy().tolist()]
                capsules = [memory_pool[i] for i in userid.flatten().numpy().tolist()]
                caps, create = basecapnet(
                            user_id=userid.to(device),
                            hist=histseq.to(device),
                            tgt=tgtseq.to(device),
                            histlen=histlen.to(device),
                            tgtlen=tgtlen.to(device),
                            caps=capsules,
                            Ks=Ks,
                            oldKs=old_Ks,
                            proj=proj_btn,
                            hist_mask=histseq_mask.to(device),
                            tgt_mask=tgtseq_mask.to(device),
                            memflag=True
                )
                # print(create)
                # print(Ks)
            if prune_btn:
                for j,user in enumerate(userid):
                    user = user.item()
                    for K_id in range(old_Ks[j]):
                        memory_pool[user][K_id] = caps[j, K_id].detach()
                    for K_id in range(Ks[j]-1,old_Ks[j]-1,-1):
                        temp_caps = caps[j,K_id].detach()
                        if torch.norm(temp_caps)<0.01:
                            memory_pool[user].pop(K_id)
                            memory_len[user] -= 1
                        else:
                            memory_pool[user][K_id] = caps[j, K_id].detach()
            else:
                for j,user in enumerate(userid):
                    user = user.item()
                    for K_id in range(Ks[j]):
                        memory_pool[user][K_id] = caps[j,K_id].detach()

        #evaluation
        #testing on the subsequent items based on the memoried capsules
        total_test_loss = []
        testdataloader = DataLoader(Traintask(trainset_split,idx+1),shuffle=False,batch_size=batchsize)
        for i, (userid,histseq,tgtseq,histlen,tgtlen,histseq_mask,tgtseq_mask) in enumerate(testdataloader):
            Ks = [memory_len[i] for i in userid.flatten().numpy().tolist()]
            capsules = [memory_pool[i] for i in userid.flatten().numpy().tolist()]
            pos_loss, neg_loss = basecapnet(
                    user_id=userid.to(device),
                    hist=histseq.to(device),
                    tgt=histseq.to(device),
                    histlen=histlen.to(device),
                    tgtlen=histlen.to(device),
                    caps=capsules,
                    Ks=Ks,
                    oldKs = Ks,
                    proj=False,
                    hist_mask=histseq_mask.to(device),
                    tgt_mask=histseq_mask.to(device),
                    memflag=False
            )
            loss = -(torch.sum(pos_loss) + torch.sum(neg_loss))/(1+nsr)/(torch.sum(histlen))
            total_test_loss.append(loss.item())
        print(sum(total_test_loss)/len(total_test_loss))

    



