"""
Let's define the two kind of architectures:

##################################################
# ARCHITECTURE 1
##################################################

                                   ↑
                           ModalitiesFusionArc1
                                   ↑
            ┌ ------------------------------------------ ┐
            ↑                      ↑                     ↑
    SlowFastFusionArc1     SlowFastFusionArc1     SlowFastFusionArc1
            ↑                      ↑                     ↑
       ┌ ------- ┐            ┌ -------- ┐           ┌ ------- ┐
       ↑         ↑            ↑          ↑           ↑         ↑   
    RGB-Slow  RGB-Fast     Obj-Slow  Obj-Fast     Flow-Slow  Flow-Fast


##################################################
# ARCHITECTURE 2
##################################################

                                   ↑
                           SlowFastFusionArch2
                                   ↑
                 ┌ -------------------------------- ┐
                 ↑                                  ↑
       ModalitiesFusionArc2               ModalitiesFusionArc2
                 ↑                                  ↑
       ┌ ----------------- ┐              ┌ ----------------- ┐
       ↑         ↑         ↑              ↑         ↑         ↑
    RGB-Slow  Obj-Slow  Flow-Slow      RGB-Fast  Obj-Fast  Flow-Fast
"""

from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F

import pytorch_lightning as pl

class OpenLSTM(nn.Module):
    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        super(OpenLSTM, self).__init__()
        self.lstm = nn.LSTM(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        last_cell = None
        last_hid = None
        hid = []
        cell = []
        for i in range(seq.shape[0]):
            el = seq[i, ...].unsqueeze(0)
            if last_cell is not None:
                _, (last_hid, last_cell) = self.lstm(el, (last_hid, last_cell))
            else:
                _, (last_hid, last_cell) = self.lstm(el)
            hid.append(last_hid)
            cell.append(last_cell)
        return torch.stack(hid, 0),  torch.stack(cell, 0)

class RULSTM(pl.LightningModule):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1, sequence_completion=False, 
                 return_context=False):
        super().__init__()
        self.feat_in = feat_in
        self.h_dim = hidden
        self.sequence_completion = sequence_completion
        self.return_context = return_context

        self.dropout = nn.Dropout(dropout)
        self.rolling_lstm = OpenLSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.unrolling_lstm = nn.LSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden, num_class)
        )

    def forward(self, inputs):
        B, T, _ = inputs.shape

        inputs = inputs.permute(1, 0, 2) # [T, B, feat_in]
        x, c = self.rolling_lstm(self.dropout(inputs))
        x = x.contiguous() # [B, T, h_dim]
        c = c.contiguous() # [B, T, h_dim]
        predictions = [] # accumulate the predictions in a list
        for t in range(T):
            hid = x[t, ...]
            cel = c[t, ...]
            if self.sequence_completion:
                ins = inputs[t:, ...]
            else:
                ins = inputs[t, ...].unsqueeze(0).expand(T - t + 1, B, self.feat_in)
            h_t, (_,_) = self.unrolling_lstm(self.dropout(ins), (hid.contiguous(), cel.contiguous()))
            h_n = h_t[-1, ...]
            predictions += [h_n]

        x = torch.stack(predictions, 1) # [B, T, h_dim]
        logits = self.classifier(x.view(-1, self.h_dim)).view(B, T, -1)
        if self.return_context:
            c = c.squeeze().permute(1, 0, 2) # [T, B, D]
            return logits, torch.cat([x, c], 2)
        else:
            return logits

class ModalitiesFusionArc2(nn.Module):
    def __init__(self, branches, hidden, dropout=0.8, slow_fast_fusion_size=1, return_context=False):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            hidden: size of hidden vectors of the branches
            dropout: dropout probability
        """
        super(RULSTMFusion, self).__init__()
        self.branches = nn.ModuleList(branches)
        self.return_context = return_context

        # input size for the MATT network
        # given by 2 (hidden and cell state) * num_branches * hidden_size
        slow_fast_fusion_size = 1
        in_size = slow_fast_fusion_size*2*len(self.branches)*hidden
        
        # MATT network: an MLP with 3 layers
        self.MATT = nn.Sequential(nn.Linear(in_size,int(in_size/4)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/4), int(in_size/8)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/8), len(self.branches)))
        #self.MATT = nn.Sequential(nn.Linear(in_size, len(self.branches)))
        #self.classifier = nn.Sequential(nn.Linear(2513*3, 2513))


    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches"""
        scores, contexts = [], []

        # for each branch
        for i in range(len(inputs)):
            # feed the inputs to the LSTM and get the scores and context vectors
            #print(i, inputs[i].shape)
            #with torch.no_grad():
            #print(inputs[i].shape, "1111")
            s, c = self.branches[i](inputs[i])
            #s = self.branches[i](inputs[i])
            """if(c.shape[1] == 24):
                idxs = p.arange(12, 24, 2)
                s = s[:, idxs]
                c = c[:, idxs]"""
            #for j in range(len(s)): 
            #    scores.append(s[j])
            scores.append(s)
            contexts.append(c)

        #print(contexts[0].shape)
        context = torch.cat(contexts, 2)
        context = context.view(-1, context.shape[-1])

        # Apply the MATT network to the context vectors
        # and normalize the outputs using softmax
        a = F.softmax(self.MATT(context),1)
        #a = context
        #print(a.shape) #, contexts[0].shape, context.shape, len(scores), scores[0].shape)
      
        """#pred_shape = scores[0].shape
        scores_in = torch.cat(scores, 2)
        scores_in = scores_in.view(-1, scores_in.shape[-1])
        a = F.softmax(self.MATT(scores_in),1)
        #p = self.classifier(scores)
        #p = p.view(pred_shape)
        #print(p.shape, pred_shape)"""

        # array to contain the fused scores
        sc = torch.zeros_like(scores[0])

        # fuse all scores multiplying by the weights
        for i in range(len(scores)): #len(inputs)):
            s = (scores[i].view(-1,scores[i].shape[-1])*a[:,i].unsqueeze(1)).view(sc.shape)
            sc += s

        if(self.return_context):
            c = torch.zeros_like(contexts[0])
            for i in range(len(inputs)):
                c += (contexts[i].view(-1,contexts[i].shape[-1])*a[:,i].unsqueeze(1)).view(c.shape)
            #c = attention.view([contexts[0].shape[0], contexts[0].shape[1], 2])
            #c = context = torch.cat(contexts, 2)
            #print(c.shape)
            #c = self.MATT[:-1](context)
            #c = c.view([contexts[0].shape[0], contexts[0].shape[1], 768])
            #print(c.shape)
            return sc, c
        else:
            # return the fused scores
            return sc
            #return p

class ModalitiesFusionArc1(nn.Module):
    def __init__(self, branches, hidden, dropout=0.8, slow_fast_fusion_size=1, return_context=False):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            hidden: size of hidden vectors of the branches
            dropout: dropout probability
        """
        super(AllBranchesRULSTMFusion, self).__init__()
        self.branches = nn.ModuleList(branches)
        self.return_context = return_context

        # input size for the MATT network
        # given by 2 (hidden and cell state) * num_branches * hidden_size
        slow_fast_fusion_size = 1
        in_size = slow_fast_fusion_size*2*len(self.branches)*hidden
        
        # MATT network: an MLP with 3 layers
        self.MATT = nn.Sequential(nn.Linear(in_size,int(in_size/4)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/4), int(in_size/8)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/8), len(self.branches)))
        #self.MATT = nn.Sequential(nn.Linear(in_size, len(self.branches)))
        #self.classifier = nn.Sequential(nn.Linear(2513*3, 2513))


    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches"""
        scores, contexts = [], []

        # for each branch
        for i in range(len(inputs)):
            # feed the inputs to the LSTM and get the scores and context vectors
            #print(i, inputs[i].shape)
            #with torch.no_grad():
            #print(inputs[i].shape, "1111")
            s, c = self.branches[i](inputs[i])
            #s = self.branches[i](inputs[i])
            """if(c.shape[1] == 24):
                idxs = p.arange(12, 24, 2)
                s = s[:, idxs]
                c = c[:, idxs]"""
            #for j in range(len(s)): 
            #    scores.append(s[j])
            scores.append(s)
            contexts.append(c)

        #print(contexts[0].shape)
        context = torch.cat(contexts, 2)
        context = context.view(-1, context.shape[-1])

        # Apply the MATT network to the context vectors
        # and normalize the outputs using softmax
        a = F.softmax(self.MATT(context),1)
        #a = context
        #print(a.shape) #, contexts[0].shape, context.shape, len(scores), scores[0].shape)
      
        """#pred_shape = scores[0].shape
        scores_in = torch.cat(scores, 2)
        scores_in = scores_in.view(-1, scores_in.shape[-1])
        a = F.softmax(self.MATT(scores_in),1)
        #p = self.classifier(scores)
        #p = p.view(pred_shape)
        #print(p.shape, pred_shape)"""

        # array to contain the fused scores
        sc = torch.zeros_like(scores[0])

        # fuse all scores multiplying by the weights
        for i in range(len(scores)): #len(inputs)):
            s = (scores[i].view(-1,scores[i].shape[-1])*a[:,i].unsqueeze(1)).view(sc.shape)
            sc += s

        if(self.return_context):
            c = torch.zeros_like(contexts[0])
            for i in range(len(inputs)):
                c += (contexts[i].view(-1,contexts[i].shape[-1])*a[:,i].unsqueeze(1)).view(c.shape)
            #c = attention.view([contexts[0].shape[0], contexts[0].shape[1], 2])
            #c = context = torch.cat(contexts, 2)
            #print(c.shape)
            #c = self.MATT[:-1](context)
            #c = c.view([contexts[0].shape[0], contexts[0].shape[1], 768])
            #print(c.shape)
            return sc, c
        else:
            # return the fused scores
            return sc
            #return p


class SlowFastFusionArc2(nn.Module):
    def __init__(self, branches, hidden, dropout=0.8, alphas=[0.125, 0.5], return_context=False):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            hidden: size of hidden vectors of the branches
            dropout: dropout probability
            steps: steps between different frame rates
        """
        super(RULSTMSlowFastFusion, self).__init__()
        self.branches = nn.ModuleList(branches)
        self.alphas = alphas
        self.return_context = return_context

        # input size for the MATT network
        # given by 2 (hidden and cell state) * num_branches * hidden_size
        in_size = 2*len(self.branches)*hidden
        
        # MATT network: an MLP with 3 layers
        self.MATT = nn.Sequential(nn.Linear(in_size,int(in_size/4)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/4), int(in_size/8)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/8), len(self.branches)))


    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches"""
        scores, slow_scores, contexts = [], [], []
        #max_step = max(self.steps)
        # for each branch
        for i in range(len(inputs)):
            #print(len(inputs[i]))
            if(self.alphas[i] != 0.5):
                inputs[i][0] = inputs[i][0][:, 12:]
                inputs[i][1] = inputs[i][1][:, 12:]
                inputs[i][2] = inputs[i][2][:, 12:]
            if i>0:
                step = int(self.alphas[i]/min(self.alphas))
                input = torch.zeros([inputs[i][0].shape[0], int(inputs[i][0].shape[1]/step), inputs[i][0].shape[2]])
                input1 = torch.zeros([inputs[i][1].shape[0], int(inputs[i][1].shape[1]/step), inputs[i][1].shape[2]])
                input2 = torch.zeros([inputs[i][2].shape[0], int(inputs[i][2].shape[1]/step), inputs[i][2].shape[2]])
                input[:,-int(1.0/self.alphas[i])] = inputs[i][0][:, -int(1.0/min(self.alphas))]
                input1[:,-int(1.0/self.alphas[i])] = inputs[i][1][:, -int(1.0/min(self.alphas))]
                input2[:,-int(1.0/self.alphas[i])] = inputs[i][2][:, -int(1.0/min(self.alphas))]
                counter = 1
                for j in range(-int(1.0/self.alphas[i])-1,-input.shape[1]-1, -1):
                    input[:, j] = inputs[i][0][:, -int(1.0/min(self.alphas))-(counter*step)]
                    input1[:, j] = inputs[i][1][:, -int(1.0/min(self.alphas))-(counter*step)]
                    input2[:, j] = inputs[i][2][:, -int(1.0/min(self.alphas))-(counter*step)]
                    #print(-int(1.0/self.alphas[i]), -int(1.0/min(self.alphas)), j,  -int(1.0/min(self.alphas))-(counter*step))
                    counter += 1
                counter = 1
                for j in range(-int(1.0/self.alphas[i])+1,0):
                    input[:, j] = inputs[i][0][:, -int(1.0/min(self.alphas))+(counter*step)]
                    input1[:, j] = inputs[i][1][:, -int(1.0/min(self.alphas))+(counter*step)]
                    input2[:, j] = inputs[i][2][:, -int(1.0/min(self.alphas))+(counter*step)]
                    #print(-int(1.0/self.alphas[i]), -int(1.0/min(self.alphas)), j,  -int(1.0/min(self.alphas))+(counter*step), "****")
                    counter += 1
                inputs[i][0] = input.to(inputs[i][0].device)
                inputs[i][1] = input1.to(inputs[i][1].device)
                inputs[i][2] = input2.to(inputs[i][2].device) 
                #inputs[i] = inputs[i][:, np.arange(step-1,inputs[i].shape[1],step)]

            # feed the inputs to the LSTM and get the scores and context vectors
            #if(i==0):
            #    with torch.no_grad(): 
            #        s, c = self.branches[i](inputs[i])
            #else:
            with torch.no_grad():
                #print(inputs[i][0].shape)
                s, c = self.branches[i](inputs[i])
            #print(self.alphas[i], s.shape, c.shape)
            #for s,c in zip(s_all, c_all):
                #print(s.shape, c.shape)
            if i < len(inputs)-1:
                step = int(max(self.alphas)/self.alphas[i])
                c_slow = torch.zeros([c.shape[0], int(c.shape[1]/step), c.shape[2]]).to(c.device)
                s_slow = torch.zeros([s.shape[0], int(s.shape[1]/step), s.shape[2]]).to(s.device)
                c_slow[:,-int(1.0/max(self.alphas))] = c[:, -int(1.0/self.alphas[i])]
                s_slow[:,-int(1.0/max(self.alphas))] = s[:, -int(1.0/self.alphas[i])]
                counter = 1
                for j in range(-int(1.0/max(self.alphas))-1,-c_slow.shape[1]-1, -1):
                    c_slow[:, j] = c[:, -int(1.0/self.alphas[i])-(counter*step)]
                    s_slow[:, j] = s[:, -int(1.0/self.alphas[i])-(counter*step)]
                    #print(self.alphas[i], -int(1.0/max(self.alphas)), -int(1.0/self.alphas[i]), j, -int(1.0/self.alphas[i])-(counter*step))
                    counter += 1
                counter = 1
                for j in range(-int(1.0/max(self.alphas))+1,0):
                    c_slow[:, j] = c[:, -int(1.0/self.alphas[i])+(counter*step)]
                    s_slow[:, j] = s[:, -int(1.0/self.alphas[i])+(counter*step)]
                    #print(self.alphas[i], -int(1.0/max(self.alphas)), -int(1.0/self.alphas[i]), j, -int(1.0/self.alphas[i])+(counter*step), "****")
                    counter += 1
                c = c_slow
                #c = c[:, np.arange(step-1,c.shape[1],step)]
                #s_slow = s[:, np.arange(step-1,s.shape[1],step)]
            else:
                s_slow = s[:, 3:].contiguous()
                c = c[:, 3:].contiguous()

            #print(inputs[i].shape, c.shape, s_slow.shape)
            #print(s.shape, c.shape)
            scores.append(s)
            #if(len(scores) == 1 or len(scores) == 4):
            contexts.append(c)
            slow_scores.append(s_slow)
            #contexts.append(c)
            #print(c.shape, s_slow.shape)
        #print(len(contexts), contexts[0].shape)
        context = torch.cat(contexts, 2)
        #print(context.shape)
        context = context.view(-1, context.shape[-1])

        # Apply the MATT network to the context vectors
        # and normalize the outputs using softmax
        #attention = self.MATT(context)
        #print(self.MATT[:-1](context).shape)
        #with torch.no_grad():
        a = F.softmax(self.MATT(context),1)
        #print(a.shape, "before return")

        # array to contain the fused scores
        sc = torch.zeros_like(slow_scores[0])

        # fuse all scores multiplying by the weights
        for i in range(len(slow_scores)):
            #print(slow_scores[i].shape, a[:,i].shape, slow_scores[i].shape[-1])
            s = (slow_scores[i].view(-1,slow_scores[i].shape[-1])*a[:,i].unsqueeze(1)).view(sc.shape)
            #print(s.shape)
            sc += s
        """sc = slow_scores[0].clone()
        for i in range(sc.shape[0]):
            corr = np.corrcoef(inputs[0][i].cpu().numpy())
            avrg_corr = np.mean(np.tril(abs(corr), -1))
            if(avrg_corr >= 0.35):
                sc[i] = slow_scores[1][i].clone()"""
        """for i in range(scores[0].shape[1]):
            if i%max_step == 0:
                scores[0][:,i] = sc[:,int((i+1)/max_step)-1]"""

        if(self.return_context):
            c = torch.zeros_like(contexts[0])
            for i in range(len(inputs)):
                c += (contexts[i].view(-1,contexts[i].shape[-1])*a[:,i].unsqueeze(1)).view(c.shape)
            #c = attention.view([contexts[0].shape[0], contexts[0].shape[1], 2])
            #c = context = torch.cat(contexts, 2)
            return sc, c
        else:
            # return the fused scores
            return sc



class SlowFastFusionArc1(nn.Module):
    def __init__(self, branches, hidden, dropout=0.8, alphas=[0.125, 0.5], return_context=False):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            hidden: size of hidden vectors of the branches
            dropout: dropout probability
            steps: steps between different frame rates
        """
        super(SingleBranchRULSTMSlowFastFusion, self).__init__()
        self.branches = nn.ModuleList(branches)
        self.alphas = alphas
        self.return_context = return_context

        # input size for the MATT network
        # given by 2 (hidden and cell state) * num_branches * hidden_size
        in_size = 2*len(self.branches)*hidden
        
        # MATT network: an MLP with 3 layers
        self.MATT = nn.Sequential(nn.Linear(in_size,int(in_size/4)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/4), int(in_size/8)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/8), len(self.branches)))


    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches"""
        scores, slow_scores, contexts = [], [], []
        #max_step = max(self.steps)
        # for each branch
        for i in range(len(inputs)):
            #print(len(inputs[i]))
            if(self.alphas[i] != 0.5):
                inputs[i] = inputs[i][:, 12:]
            if i>0:
                step = int(self.alphas[i]/min(self.alphas))
                input = torch.zeros([inputs[i].shape[0], int(inputs[i].shape[1]/step), inputs[i].shape[2]])
                input[:,-int(1.0/self.alphas[i])] = inputs[i][:, -int(1.0/min(self.alphas))]
                counter = 1
                for j in range(-int(1.0/self.alphas[i])-1,-input.shape[1]-1, -1):
                    input[:, j] = inputs[i][:, -int(1.0/min(self.alphas))-(counter*step)]
                    #print(-int(1.0/self.alphas[i]), -int(1.0/min(self.alphas)), j,  -int(1.0/min(self.alphas))-(counter*step))
                    counter += 1
                counter = 1
                for j in range(-int(1.0/self.alphas[i])+1,0):
                    input[:, j] = inputs[i][:, -int(1.0/min(self.alphas))+(counter*step)]
                    #print(-int(1.0/self.alphas[i]), -int(1.0/min(self.alphas)), j,  -int(1.0/min(self.alphas))+(counter*step), "****")
                    counter += 1
                inputs[i] = input.to(inputs[i].device)
                #inputs[i] = inputs[i][:, np.arange(step-1,inputs[i].shape[1],step)]

            # feed the inputs to the LSTM and get the scores and context vectors
            #if(i==0):
            #    with torch.no_grad(): 
            #        s, c = self.branches[i](inputs[i])
            #else:
            #with torch.no_grad():
                #print(inputs[i][0].shape)
            s, c = self.branches[i](inputs[i])
            #print(self.alphas[i], s.shape, c.shape)
            #for s,c in zip(s_all, c_all):
                #print(s.shape, c.shape)
            if i < len(inputs)-1:
                step = int(max(self.alphas)/self.alphas[i])
                c_slow = torch.zeros([c.shape[0], int(c.shape[1]/step), c.shape[2]]).to(c.device)
                s_slow = torch.zeros([s.shape[0], int(s.shape[1]/step), s.shape[2]]).to(s.device)
                c_slow[:,-int(1.0/max(self.alphas))] = c[:, -int(1.0/self.alphas[i])]
                s_slow[:,-int(1.0/max(self.alphas))] = s[:, -int(1.0/self.alphas[i])]
                counter = 1
                for j in range(-int(1.0/max(self.alphas))-1,-c_slow.shape[1]-1, -1):
                    c_slow[:, j] = c[:, -int(1.0/self.alphas[i])-(counter*step)]
                    s_slow[:, j] = s[:, -int(1.0/self.alphas[i])-(counter*step)]
                    #print(self.alphas[i], -int(1.0/max(self.alphas)), -int(1.0/self.alphas[i]), j, -int(1.0/self.alphas[i])-(counter*step))
                    counter += 1
                counter = 1
                for j in range(-int(1.0/max(self.alphas))+1,0):
                    c_slow[:, j] = c[:, -int(1.0/self.alphas[i])+(counter*step)]
                    s_slow[:, j] = s[:, -int(1.0/self.alphas[i])+(counter*step)]
                    #print(self.alphas[i], -int(1.0/max(self.alphas)), -int(1.0/self.alphas[i]), j, -int(1.0/self.alphas[i])+(counter*step), "****")
                    counter += 1
                c = c_slow
                #c = c[:, np.arange(step-1,c.shape[1],step)]
                #s_slow = s[:, np.arange(step-1,s.shape[1],step)]
            else:
                s_slow = s[:, 3:].contiguous()
                c = c[:, 3:].contiguous()

            #print(inputs[i].shape, c.shape, s_slow.shape)
            #print(s.shape, c.shape)
            scores.append(s)
            #if(len(scores) == 1 or len(scores) == 4):
            contexts.append(c)
            slow_scores.append(s_slow)
            #contexts.append(c)
            #print(c.shape, s_slow.shape)
        #print(len(contexts), contexts[0].shape)
        context = torch.cat(contexts, 2)
        #print(context.shape)
        context = context.view(-1, context.shape[-1])

        # Apply the MATT network to the context vectors
        # and normalize the outputs using softmax
        #attention = self.MATT(context)
        #print(self.MATT[:-1](context).shape)
        #with torch.no_grad():
        a = F.softmax(self.MATT(context),1)
        #print(a.shape, "before return")

        # array to contain the fused scores
        sc = torch.zeros_like(slow_scores[0])

        # fuse all scores multiplying by the weights
        for i in range(len(slow_scores)):
            #print(slow_scores[i].shape, a[:,i].shape, slow_scores[i].shape[-1])
            s = (slow_scores[i].view(-1,slow_scores[i].shape[-1])*a[:,i].unsqueeze(1)).view(sc.shape)
            #print(s.shape)
            sc += s
        """sc = slow_scores[0].clone()
        for i in range(sc.shape[0]):
            corr = np.corrcoef(inputs[0][i].cpu().numpy())
            avrg_corr = np.mean(np.tril(abs(corr), -1))
            if(avrg_corr >= 0.35):
                sc[i] = slow_scores[1][i].clone()"""
        """for i in range(scores[0].shape[1]):
            if i%max_step == 0:
                scores[0][:,i] = sc[:,int((i+1)/max_step)-1]"""

        if(self.return_context):
            c = torch.zeros_like(contexts[0])
            for i in range(len(inputs)):
                c += (contexts[i].view(-1,contexts[i].shape[-1])*a[:,i].unsqueeze(1)).view(c.shape)
            #c = attention.view([contexts[0].shape[0], contexts[0].shape[1], 2])
            #c = context = torch.cat(contexts, 2)
            return sc, c
        else:
            # return the fused scores
            return sc
