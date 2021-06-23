from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F

class OpenLSTM(nn.Module):
    """"An LSTM implementation that returns the intermediate hidden and cell states.
    The original implementation of PyTorch only returns the last cell vector.
    For RULSTM, we want all cell vectors computed at intermediate steps"""
    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        """
            feat_in: input feature size
            feat_out: output feature size
            num_layers: number of layers
            dropout: dropout probability
        """
        super(OpenLSTM, self).__init__()

        # simply create an LSTM with the given parameters
        self.lstm = nn.LSTM(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        # manually iterate over each input to save the individual cell vectors
        last_cell=None
        last_hid=None
        hid = []
        cell = []
        for i in range(seq.shape[0]):
            el = seq[i,...].unsqueeze(0)
            if last_cell is not None:
                _, (last_hid, last_cell) = self.lstm(el, (last_hid,last_cell))
            else:
                _, (last_hid, last_cell) = self.lstm(el)
            hid.append(last_hid)
            cell.append(last_cell)

        return torch.stack(hid, 0),  torch.stack(cell, 0)

class RULSTM(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1, 
            sequence_completion=False, return_context=False):
        """
            num_class: number of classes
            feat_in: number of input features
            hidden: number of hidden units
            dropout: dropout probability
            depth: number of LSTM layers
            sequence_completion: if the network should be arranged for sequence completion pre-training
            return_context: whether to return the Rolling LSTM hidden and cell state (useful for MATT) during forward
        """
        super(RULSTM, self).__init__()
        self.feat_in = feat_in
        self.dropout = nn.Dropout(dropout)
        self.hidden=hidden
        self.rolling_lstm = OpenLSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.unrolling_lstm = nn.LSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        """self.fast_unrolling_lstm = nn.LSTM(feat_in+hidden, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)"""
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_class))
        self.sequence_completion = sequence_completion
        self.return_context = return_context

        self.fast_slow_fusion = nn.Sequential(nn.Dropout(dropout), nn.Linear(2*hidden,hidden), nn.ReLU())

        """#attention to wait past predictions(28 for 0.125 alpha for trial)
        in_size = 2*28*hidden
        self.MATT = nn.Sequential(nn.Linear(in_size,int(in_size/4)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/4), int(in_size/8)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/8), 28))"""

    def forward(self, inputs):
        # permute the inputs for compatibility with the LSTM
        step = 4 #the step to generate the slow rate
        #print(inputs.shape, "2222")
        inputs=inputs.permute(1,0,2)
        """inputs_slow = inputs[np.arange(step-1,inputs.shape[0],step)]"""

        """inputs_slow = torch.zeros([int(inputs.shape[0]/step), inputs.shape[1], inputs.shape[2]])
        inputs_slow[-2] = inputs[-8]
        counter = 1
        for j in range(-3,-inputs_slow.shape[0]-1, -1):
            inputs_slow[j] = inputs[-8-(counter*step)]
            #print(j,  -8-(counter*step))
            counter += 1
        counter = 1
        for j in range(-1,0):
            inputs_slow[j] = inputs[-8+(counter*step)]
            #print(j,  -8+(counter*step), "****")
            counter += 1
        inputs_slow = inputs_slow.to(inputs.device)"""


        # pass the frames through the rolling LSTM
        # and get the hidden (x) and cell (c) states at each time-step
        x, c = self.rolling_lstm(self.dropout(inputs))
        x = x.contiguous() # batchsize x timesteps x hidden
        c = c.contiguous() # batchsize x timesteps x hidden

        #hidden state attention
        #in_state = torch.cat([x, c],2)
        #a = F.softmax(self.MATT(in_state),1)
        #print(a.shape)

        """x_slow, c_slow = self.rolling_lstm(self.dropout(inputs_slow))
        x_slow = x_slow.contiguous() # batchsize x timesteps x hidden
        c_slow = c_slow.contiguous() # batchsize x timesteps x hidden"""

        # accumulate the predictions in a list
        predictions = [] # accumulate the predictions in a list
        #predictions_slow = []

        # for each time-step
        for t in range(x.shape[0]):
            """if(t%step == 0):
                # get the hidden and cell states at current time-step
                hid_slow = x_slow[int(t/step),...]
                cel_slow = c_slow[int(t/step),...]

                if self.sequence_completion:
                    # take current + future inputs (looks into the future)
                    ins_slow = inputs_slow[int(t/step):,...]
                else:
                    # replicate the current input for the correct number of times (time-steps remaining to the beginning of the action)
                    ins_slow = inputs_slow[int(t/step),...].unsqueeze(0).expand(inputs_slow.shape[0]-(int(t/step))+1,
                               inputs_slow.shape[1],inputs_slow.shape[2]).to(inputs_slow.device)

                # initialize the LSTM and iterate over the inputs
                h_t_slow, (_,_) = self.unrolling_lstm(self.dropout(ins_slow), (hid_slow.contiguous(), cel_slow.contiguous()))
                # get last hidden state
                h_n_slow = h_t_slow[-1,...]

                #predictions_slow.append(h_n_slow)"""


            # get the hidden and cell states at current time-step
            hid = x[t,...]
            cel = c[t,...]

            if self.sequence_completion:
                # take current + future inputs (looks into the future)
                ins = inputs[t:,...]
            else:
                # replicate the current input for the correct number of times (time-steps remaining to the beginning of the action)
                ins = inputs[t,...].unsqueeze(0).expand(inputs.shape[0]-t+1,inputs.shape[1],inputs.shape[2]).to(inputs.device)

            #new_ins = torch.empty((ins.shape[0], ins.shape[1], self.feat_in+self.hidden)).to(inputs.device)
            """for i in range(ins.shape[0]):
                if(i%step == 0):
                    ins[i] = h_t_slow[int(i/step)].clone()"""

            # initialize the LSTM and iterate over the inputs
            h_t, (_,_) = self.unrolling_lstm(self.dropout(ins), (hid.contiguous(), cel.contiguous()))
            # get last hidden state
            h_n = h_t[-1,...]

            # append the last hidden state to the list
            """if(t%step == 0):
                #fused_prediction = self.fast_slow_fusion(torch.cat((h_n, h_n_slow),1))
                predictions.append(torch.cat((h_n, h_n_slow),1))#fused_prediction)
            #else:"""
            predictions.append(h_n)


        # obtain the final prediction tensor by concatenating along dimension 1
        x = torch.stack(predictions,1)
        #x_slow = torch.stack(predictions_slow,1)

        # apply the classifier to each output feature vector (independently)
        y = self.classifier(x.view(-1,x.size(2))).view(x.size(0), x.size(1), -1)
        #y_slow = self.classifier(x_slow.view(-1,x_slow.size(2))).view(x_slow.size(0), x_slow.size(1), -1)

        #attention
        #for i in range(y.shape[1]):
            

        """for i in range(y.shape[1]):
            if(i%step == 0):
                y[:,i] = (0.5*y[:,i].clone()).add(y_slow[:,int(i/step)].clone()*0.5)"""
        #for i in range(y_slow.shape[1]):
        #    y_slow[:, i] = (0.5*y[:,i*step].clone()).add(y_slow[:,i].clone()*0.5)
        if self.return_context:
            # return y and the concatenation of hidden and cell states 
            c=c.squeeze().permute(1,0,2)
            return y, torch.cat([x, c],2)
        else:
            return y

class RULSTMFusion(nn.Module):
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

class AllBranchesRULSTMFusion(nn.Module):
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


class RULSTMSlowFastFusion(nn.Module):
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



class SingleBranchRULSTMSlowFastFusion(nn.Module):
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
