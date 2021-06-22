"""Main training/test program for RULSTM"""
from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
from models import RULSTM, ModalitiesFusionArc2, SlowFastFusionArc2, SlowFastFusionArc1, ModalitiesFusionArc1
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils import topk_accuracy, ValueMeter, topk_accuracy_multiple_timesteps, get_marginal_indexes, marginalize, \
        softmax,  topk_recall_multiple_timesteps, tta, predictions_to_json, MeanTopKRecallMeter
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
pd.options.display.float_format = '{:05.2f}'.format

parser = ArgumentParser(description="Training program for RULSTM")
parser.add_argument('mode', type=str, choices=['train', 'validate', 'test', 'test', 'validate_json'], default='train',
                    help="Whether to perform training, validation or test.\
                            If test is selected, --json_directory must be used to provide\
                            a directory in which to save the generated jsons.")
parser.add_argument('path_to_data', type=str,
                    help="Path to the data folder, \
                            containing all LMDB datasets")
parser.add_argument('path_to_models', type=str,
                    help="Path to the directory where to save all models")
parser.add_argument('--alpha', type=float, default=0.25,
                    help="Distance between time-steps in seconds")
parser.add_argument('--alphas_fused', type=float, nargs='+', default=[0.125,0.5])
parser.add_argument('--S_enc', type=int, default=6,
                    help="Number of encoding steps. \
                            If early recognition is performed, \
                            this value is discarded.")
parser.add_argument('--S_enc_fused', type=int, nargs='+', default=[12,3])
parser.add_argument('--S_ant', type=int, default=8,
                    help="Number of anticipation steps. \
                            If early recognition is performed, \
                            this is the number of frames sampled for each action.")
parser.add_argument('--S_ant_fused', type=int, nargs='+', default=[16,4])
parser.add_argument('--task', type=str, default='anticipation', choices=[
                    'anticipation', 'early_recognition'], help='Task to tackle: \
                            anticipation or early recognition')
parser.add_argument('--img_tmpl', type=str,
                    default='frame_{:010d}.jpg', help='Template to use to load the representation of a given frame')
parser.add_argument('--modality', type=str, default='rgb',
                    choices=['rgb', 'flow', 'obj', 'fusion'], help = "Modality. Rgb/flow/obj represent single branches, whereas fusion indicates the whole model with modality attention.")
parser.add_argument('--slowfastfusion', action='store_true')
parser.add_argument('--sequence_completion', action='store_true',
                    help='A flag to selec sequence completion pretraining rather than standard training.\
                            If not selected, a valid checkpoint for sequence completion pretraining\
                            should be available unless --ignore_checkpoints is specified')
parser.add_argument('--mt5r', action='store_true')

parser.add_argument('--num_class', type=int, default=2513,
                    help='Number of classes')
parser.add_argument('--hidden', type=int, default=1024,
                    help='Number of hidden units')
parser.add_argument('--feat_in', type=int, default=1024,
                    help='Input size. If fusion, it is discarded (see --feats_in)')
parser.add_argument('--feats_in', type=int, nargs='+', default=[1024, 1024, 352],
                    help='Input sizes when the fusion modality is selected.')
parser.add_argument('--dropout', type=float, default=0.8, help="Dropout rate")

parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
parser.add_argument('--num_workers', type=int, default=4,
                    help="Number of parallel thread to fetch the data")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")

parser.add_argument('--display_every', type=int, default=10,
                    help="Display every n iterations")
parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
parser.add_argument('--visdom', action='store_true',
                    help="Whether to log using visdom")

parser.add_argument('--ignore_checkpoints', action='store_true',
                    help='If specified, avoid loading existing models (no pre-training)')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume suspended training')

parser.add_argument('--ek100', action='store_true',
                    help="Whether to use EPIC-KITCHENS-100")

parser.add_argument('--json_directory', type=str, default = None, help = 'Directory in which to save the generated jsons.')

parser.add_argument('--ensamble', action='store_true')

parser.add_argument('--arc1', action='store_true')

args = parser.parse_args()

if args.mode == 'test' or args.mode=='validate_json':
    assert args.json_directory is not None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.task == 'anticipation':
    if args.slowfastfusion:
        exp_name = f"RULSTM-{args.task}_"
        for i in range(len(args.alphas_fused)):
            exp_name += f"{args.alphas_fused[i]}_{args.S_enc_fused[i]}_{args.S_ant_fused[i]}_"
        exp_name += f"{args.modality}"
    else:
        exp_name = f"RULSTM-{args.task}_{args.alpha}_{args.S_enc}_{args.S_ant}_{args.modality}"
else:
    exp_name = f"RULSTM-{args.task}_{args.alpha}_{args.S_ant}_{args.modality}"

if args.mt5r:
    exp_name += '_mt5r'

if args.sequence_completion:
    exp_name += '_sequence_completion'

if args.slowfastfusion:
    args.alpha = min(args.alphas_fused)
    args.S_enc = max(args.S_enc_fused)
    args.S_ant = max(args.S_ant_fused)


if args.visdom:
    # if visdom is required
    # load visdom loggers from torchent
    from torchnet.logger import VisdomPlotLogger, VisdomSaver
    # define loss and accuracy logger
    visdom_loss_logger = VisdomPlotLogger('line', env=exp_name, opts={
                                          'title': 'Loss', 'legend': ['training', 'validation']})
    visdom_accuracy_logger = VisdomPlotLogger('line', env=exp_name, opts={
                                              'title': 'Top5 Acc@1s', 'legend': ['training', 'validation']})
    # define a visdom saver to save the plots
    visdom_saver = VisdomSaver(envs=[exp_name])

actions_weights = np.loadtxt('actions_weights')

def get_loader(mode, override_modality = None, split_point=1.0):
    if override_modality:
        path_to_lmdb = join(args.path_to_data, override_modality)
    else:
        if args.slowfastfusion:
            if args.modality != 'fusion':
                path_to_lmdb = [join(args.path_to_data, args.modality) for m in range(len(args.alphas_fused))]
            else:
                #path_to_lmdb = [join(args.path_to_data, m) for m in ['rgb', 'flow', 'obj', 'rgb', 'flow', 'obj']]
                if args.arc1:
                    path_to_lmdb = [[join(args.path_to_data, n) for m in range(len(args.alphas_fused))] for n in ['rgb', 'flow', 'obj']]
                else:
                    path_to_lmdb = [[join(args.path_to_data, m) for m in ['rgb', 'flow', 'obj']] for n in range(len(args.alphas_fused))]
        else:
            path_to_lmdb = join(args.path_to_data, args.modality) if args.modality != 'fusion' else [join(args.path_to_data, m) for m in ['rgb', 'flow', 'obj']]

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
        'time_step': args.alpha,
        'img_tmpl': args.img_tmpl,
        'action_samples': args.S_ant if args.task == 'early_recognition' else None,
        'past_features': args.task == 'anticipation',
        'sequence_length': args.S_enc + args.S_ant,
        'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
        'challenge': 'test' in mode,
        'split_point': split_point
    }

    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training')

def get_model():
    if args.modality != 'fusion':  # single branch
        if args.slowfastfusion:
            models = []
            #steps = []
            for i in range(len(args.alphas_fused)):
                models.append(RULSTM(args.num_class, args.feat_in, args.hidden, args.dropout, return_context = args.task=='anticipation'))
                #if(i != len(args.alphas_fused)-1):
                    #steps.append(int(args.alphas_fused[i+1]/args.alphas_fused[i]))
            if args.mode == 'train' and not args.ignore_checkpoints:
                checkpoints = []
                for i in range(len(args.alphas_fused)):
                    single_exp_name = f"RULSTM-{args.task}_{args.alphas_fused[i]}_{args.S_enc_fused[i]}_{args.S_ant_fused[i]}_{args.modality}"
                    if args.sequence_completion:
                        single_exp_name += '_sequence_completion'
                    checkpoints.append(torch.load(join(args.path_to_models+f"_{args.alphas_fused[i]}", single_exp_name+'_best.pth.tar'))['state_dict'])
                    models[i].load_state_dict(checkpoints[i])
            model = SlowFastFusionArc1(models, args.hidden, args.dropout, args.alphas_fused) 
        else:
            model = RULSTM(args.num_class, args.feat_in, args.hidden,
                           args.dropout, sequence_completion=args.sequence_completion)
            # load checkpoint only if not in sequence completion mode
            # and inf the flag --ignore_checkpoints has not been specified
            if args.mode == 'train' and not args.ignore_checkpoints and not args.sequence_completion:
                checkpoint = torch.load(join(
                    args.path_to_models, exp_name + '_sequence_completion_best.pth.tar'))['state_dict']
                model.load_state_dict(checkpoint)
    else:
        if args.slowfastfusion:
            models_rgb = []
            models_flow = []
            models_obj = []
            #models = []
            
            for i in range(len(args.alphas_fused)):
                models_rgb.append(RULSTM(args.num_class, args.feats_in[0], args.hidden, args.dropout, return_context = args.task=='anticipation'))
                models_flow.append(RULSTM(args.num_class, args.feats_in[1], args.hidden, args.dropout, return_context = args.task=='anticipation'))
                models_obj.append(RULSTM(args.num_class, args.feats_in[2], args.hidden, args.dropout, return_context = args.task=='anticipation'))
                #models.append(RULSTM(args.num_class, args.feats_in[0], args.hidden, args.dropout, return_context = args.task=='anticipation'))
                #models.append(RULSTM(args.num_class, args.feats_in[1], args.hidden, args.dropout, return_context = args.task=='anticipation'))
                #models.append(RULSTM(args.num_class, args.feats_in[2], args.hidden, args.dropout, return_context = args.task=='anticipation'))

            """if args.mode == 'train' and not args.ignore_checkpoints:
                checkpoints_rgb = []
                checkpoints_flow = []
                checkpoints_obj = []
                #checkpoints = []

                for i in range(len(args.alphas_fused)):
                    single_exp_name_rgb = f"RULSTM-{args.task}_{args.alphas_fused[i]}_{args.S_enc_fused[i]}_{args.S_ant_fused[i]}_rgb"
                    single_exp_name_flow = f"RULSTM-{args.task}_{args.alphas_fused[i]}_{args.S_enc_fused[i]}_{args.S_ant_fused[i]}_flow"
                    single_exp_name_obj = f"RULSTM-{args.task}_{args.alphas_fused[i]}_{args.S_enc_fused[i]}_{args.S_ant_fused[i]}_obj"

                    checkpoints_rgb.append(torch.load(join(args.path_to_models+f"_{args.alphas_fused[i]}", single_exp_name_rgb+'_best.pth.tar'))['state_dict'])
                    checkpoints_flow.append(torch.load(join(args.path_to_models+f"_{args.alphas_fused[i]}", single_exp_name_flow+'_best.pth.tar'))['state_dict'])
                    checkpoints_obj.append(torch.load(join(args.path_to_models+f"_{args.alphas_fused[i]}", single_exp_name_obj+'_best.pth.tar'))['state_dict'])
                    #checkpoints.append(torch.load(join(args.path_to_models+f"_{args.alphas_fused[i]}", single_exp_name_rgb+'_best.pth.tar'))['state_dict'])
                    #checkpoints.append(torch.load(join(args.path_to_models+f"_{args.alphas_fused[i]}", single_exp_name_flow+'_best.pth.tar'))['state_dict'])
                    #checkpoints.append(torch.load(join(args.path_to_models+f"_{args.alphas_fused[i]}", single_exp_name_obj+'_best.pth.tar'))['state_dict'])

              
                    models_rgb[i].load_state_dict(checkpoints_rgb[i])
                    models_flow[i].load_state_dict(checkpoints_flow[i])
                    models_obj[i].load_state_dict(checkpoints_obj[i])
                    #models[i*3].load_state_dict(checkpoints[i*3])
                    #models[i*3+1].load_state_dict(checkpoints[i*3+1])
                    #models[i*3+2].load_state_dict(checkpoints[i*3+2])"""

            if args.arc1:
                rgb_model = SlowFastFusionArc1(models_rgb, args.hidden, args.dropout, args.alphas_fused, True)
                flow_model = SlowFastFusionArc1(models_flow, args.hidden, args.dropout, args.alphas_fused, True)
                obj_model = SlowFastFusionArc1(models_obj, args.hidden, args.dropout, args.alphas_fused, True)

                if args.mode == 'train' and not args.ignore_checkpoints:
                     checkpoint_rgb = torch.load(join(args.path_to_models,\
                            exp_name.replace('fusion','rgb') +'_best.pth.tar'))['state_dict']
                     checkpoint_flow = torch.load(join(args.path_to_models,\
                            exp_name.replace('fusion','flow') +'_best.pth.tar'))['state_dict']
                     checkpoint_obj = torch.load(join(args.path_to_models,\
                            exp_name.replace('fusion','obj') +'_best.pth.tar'))['state_dict']
			  
                     rgb_model.load_state_dict(checkpoint_rgb)
                     flow_model.load_state_dict(checkpoint_flow)
                     obj_model.load_state_dict(checkpoint_obj)

                model = ModalitiesFusionArc1([rgb_model, flow_model, obj_model], args.hidden, args.dropout, len(args.alphas_fused))
                #model = ModalitiesFusionArc2(models, args.hidden, args.dropout, len(args.alphas_fused))
            else:
                fast_model = RULSTMFusion([models_rgb[0], models_flow[0], models_obj[0]], args.hidden, args.dropout, return_context=True)
                slow_model = RULSTMFusion([models_rgb[1], models_flow[1], models_obj[1]], args.hidden, args.dropout, return_context=True)

                if (args.mode == 'train' and not args.ignore_checkpoints):
                     checkpoint_fast = torch.load('./models/ek55_0.125/RULSTM-anticipation_0.125_12_16_fusion_best.pth.tar')['state_dict']
                     checkpoint_slow = torch.load('./models/ek55_0.5/RULSTM-anticipation_0.5_6_4_fusion_best.pth.tar')['state_dict']
                     fast_model.load_state_dict(checkpoint_fast)
                     slow_model.load_state_dict(checkpoint_slow)

                model = RULSTMSlowFastFusion([fast_model, slow_model], args.hidden, args.dropout, args.alphas_fused)

        else:
            rgb_model = RULSTM(args.num_class, args.feats_in[0], args.hidden, args.dropout, return_context = args.task=='anticipation')
            flow_model = RULSTM(args.num_class, args.feats_in[1], args.hidden, args.dropout, return_context = args.task=='anticipation')
            obj_model = RULSTM(args.num_class, args.feats_in[2], args.hidden, args.dropout, return_context = args.task=='anticipation')
        
            if args.task=='early_recognition' or (args.mode == 'train' and not args.ignore_checkpoints):
                checkpoint_rgb = torch.load(join(args.path_to_models,\
                        exp_name.replace('fusion','rgb') +'_best.pth.tar'))['state_dict']
                checkpoint_flow = torch.load(join(args.path_to_models,\
                        exp_name.replace('fusion','flow') +'_best.pth.tar'))['state_dict']
                checkpoint_obj = torch.load(join(args.path_to_models,\
                        exp_name.replace('fusion','obj') +'_best.pth.tar'))['state_dict']

                rgb_model.load_state_dict(checkpoint_rgb)
                flow_model.load_state_dict(checkpoint_flow)
                obj_model.load_state_dict(checkpoint_obj)
        
            if args.task == 'early_recognition':
                return [rgb_model, flow_model, obj_model]

            model = RULSTMFusion([rgb_model, flow_model, obj_model], args.hidden, args.dropout)
    return model


def load_checkpoint(model, best=False):
    if best:
        chk = torch.load(join(args.path_to_models, exp_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(args.path_to_models, exp_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']
    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(model, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.path_to_models, exp_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.path_to_models, exp_name + '_best.pth.tar'))

    if args.visdom:
        # save visdom logs for persitency
        visdom_saver.save()


def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")

    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {accuracy_meter.value():.2f}% ", end="")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

    print('\033[0m')

    if args.visdom:
        visdom_loss_logger.log(epoch, loss_meter.value(), name=mode)
        visdom_accuracy_logger.log(epoch, accuracy_meter.value(), name=mode)

def get_scores_early_recognition_fusion(models, loaders):
    verb_scores = 0
    noun_scores = 0
    action_scores = 0
    for model, loader in zip(models, loaders):
        outs = get_scores(model, loader)
        verb_scores += outs[0]
        noun_scores += outs[1]
        action_scores += outs[2]

    verb_scores /= len(models)
    noun_scores /= len(models)
    action_scores /= len(models)

    return [verb_scores, noun_scores, action_scores] + list(outs[3:])


def get_scores(model, loader, challenge=False, include_discarded = False):
    model.eval()
    predictions = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['past_features' if args.task ==
                      'anticipation' else 'action_features']
            if type(x) == list:
                if(type(x[0]) == list):
                    x = [[xxx.to(device) for xxx in xx] for xx in x]
                else:
                    x = [xx.to(device) for xx in x]
            else:
                x = x.to(device)

            y = batch['label'].numpy()

            ids.append(batch['id'])
             
            """#for the average and concatenation part only (0.125 - 0.5)
            args.S_ant = 4
            args.S_enc = 3
            args.alpha =  0.5"""

            if args.slowfastfusion:
                args.S_ant = min(args.S_ant_fused)
                args.S_enc = min(args.S_enc_fused)
                args.alpha = max(args.alphas_fused)

            preds = model(x).cpu().numpy()[:, -args.S_ant:, :]
            #print( model(x).cpu().numpy()[0, :, 0], preds[0, :, 0])

            """if(models != 0):
                for i in range(preds.shape[0]):
                    corr = np.corrcoef(x[i].cpu().numpy())
                    avrg_corr = np.mean(np.tril(abs(corr), -1))
                    if(models == 1 and avrg_corr >= 0.35 or models == 2 and avrg_corr >= 0.29):
                        preds[i] = np.zeros(preds[i].shape)
                        #print(preds[i])"""

            predictions.append(preds)
            labels.append(y)

    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    actions = pd.read_csv(
        join(args.path_to_data, 'actions.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')

    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)

    if include_discarded:
        dlab = np.array(loader.dataset.discarded_labels)
        dislab = np.array(loader.dataset.discarded_ids)
        ids = np.concatenate([ids, dislab])
        num_disc = len(dlab)
        labels = np.concatenate([labels, dlab])
        verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
        noun_scores = np.concatenate((noun_scores, np.zeros((num_disc, *noun_scores.shape[1:]))))
        action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

    if labels.max()>0 and not challenge:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2], ids
    else:
        return verb_scores, noun_scores, action_scores, ids


def trainval(model, loaders, optimizer, epochs, start_epoch, start_best_perf):
    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    #scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.99)
    stop = 0
    for epoch in range(start_epoch, epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        if args.mt5r:
            accuracy_meter = {'training': MeanTopKRecallMeter(args.num_class), 'validation': MeanTopKRecallMeter(args.num_class)}
        else:
            accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                #import random as rn
                #b = rn.randint(0,len(loaders[mode]))
                counter = 0
                for i, batch in enumerate(loaders[mode]):
                    #print(i, b, len(loaders[mode]))
                    #if(i!=stop and mode == 'training'):
                    #    continue
                    x = batch['past_features' if args.task ==
                              'anticipation' else 'action_features']

                    if type(x) == list:
                        if(type(x[0]) == list):
                            x = [[xxx.to(device) for xxx in xx] for xx in x]
                        else:
                            x = [xx.to(device) for xx in x]
                    else:
                        x = x.to(device)

                    y = batch['label'].to(device)

                    bs = y.shape[0]  # batch size
                    #print(x[0][0].device, "just before predict")
                    preds = model(x)
                    
                    """#for the average and concatenation part only (0.125 - 0.5)
                    args.S_ant = 4
                    args.S_enc = 3
                    args.alpha = 0.5"""
 
                    if(args.slowfastfusion):
                        #max_step = int(max(args.alphas_fused)/min(args.alphas_fused))
                        #print(preds.shape, max_step)
                        #preds = preds[:, np.arange(max_step-1, preds.shape[1], max_step)]
                        args.S_ant = min(args.S_ant_fused)
                        args.S_enc = min(args.S_enc_fused)
                        args.alpha = max(args.alphas_fused)
                        #print(preds.shape)

                    # take only last S_ant predictions
                    #print(preds[0, :, 0], preds[0, -args.S_ant:, 0])
                    preds = preds[:, -args.S_ant:, :].contiguous()

                    # linearize predictions
                    linear_preds = preds.view(-1, preds.shape[-1])
                    # replicate the labels across timesteps and linearize
                    linear_labels = y.view(-1, 1).expand(-1,
                                                         preds.shape[1]).contiguous().view(-1)

                    #ops = F.CrossEntropyFuncOptions().weight(torch.tensor(actions_weights, dtype=torch.float))
                    loss = F.cross_entropy(linear_preds, linear_labels) #, weight=torch.tensor(actions_weights, dtype=torch.float).to(device))
                    # get the predictions for anticipation time = 1s (index -4) (anticipation)
                    # or for the last time-step (100%) (early recognition)
                    # top5 accuracy at 1s
                    idx = int(1.0/args.alpha) if args.task == 'anticipation' else -1
                    # use top-5 for anticipation and top-1 for early recognition
                    k = 5 if args.task == 'anticipation' else 1
                    acc = topk_accuracy(
                        preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0]*100

                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    if args.mt5r:
                        accuracy_meter[mode].add(preds[:, idx, :].detach().cpu().numpy(),
                                                 y.detach().cpu().numpy())
                    else:
                        accuracy_meter[mode].add(acc, bs)

                    # if in training mode
                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        #scheduler.step()
                        optimizer.step()
                        #print(scheduler.get_lr())

                    # compute decimal epoch for logging
                    e = epoch + i/len(loaders[mode])

                    # log training during loop
                    # avoid logging the very first batch. It can be biased.
                    #if mode == 'training' and i != 0 and i % args.display_every == 0:
                    if i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode], accuracy_meter[mode])
                        counter += 1
                    #print(epoch, i)
                    #if mode == 'training': #and counter == stop:
                        #print(i)
                    #    break

                #if mode == 'validation' and epoch%10 == 0:
                #    scheduler.step() #(accuracy_meter[mode])
                #if mode == 'validation':
                #    scheduler2.step(accuracy_meter[mode].value())
                    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
                # log at the end of each epoch
                log(mode, epoch+1, loss_meter[mode], accuracy_meter[mode],
                    max(accuracy_meter[mode].value(), best_perf) if mode == 'validation'
                    else None, green=True)
                #if mode == 'training':
                    #break

        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(model, epoch+1, accuracy_meter['validation'].value(), best_perf,
                   is_best=is_best)
        stop += 1

def get_validation_ids():
    unseen_participants_ids = pd.read_csv(join(args.path_to_data, 'validation_unseen_participants_ids.csv'), names=['id'], squeeze=True)
    tail_verbs_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_verbs_ids.csv'), names=['id'], squeeze=True)
    tail_nouns_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_nouns_ids.csv'), names=['id'], squeeze=True)
    tail_actions_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_actions_ids.csv'), names=['id'], squeeze=True)

    return unseen_participants_ids, tail_verbs_ids, tail_nouns_ids, tail_actions_ids

def get_many_shot():
    """Get many shot verbs, nouns and actions for class-aware metrics (Mean Top-5 Recall)"""
    # read the list of many shot verbs
    many_shot_verbs = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_verbs.csv'))['verb_class'].values
    # read the list of many shot nouns
    many_shot_nouns = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_nouns.csv'))['noun_class'].values

    # read the list of actions
    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    # create the list of many shot actions
    # an action is "many shot" if at least one
    # between the related verb and noun are many shot
    many_shot_actions = []
    for a, (v, n) in a_to_vn.items():
        if v in many_shot_verbs or n in many_shot_nouns:
            many_shot_actions.append(a)

    return many_shot_verbs, many_shot_nouns, many_shot_actions


def main():
    model = get_model()
    print(type(model))
    if type(model) == list:
        model = [m.to(device) for m in model]
    else:
        model.to(device)

    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}

        if args.resume:
            start_epoch, _, start_best_perf = load_checkpoint(model)
        else:
            start_epoch = 0
            start_best_perf = 0

        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1.)

        trainval(model, loaders, optimizer, args.epochs,
                 start_epoch, start_best_perf)

    elif args.mode == 'validate':
        if args.task == 'early_recognition' and args.modality == 'fusion':
            loaders = [get_loader('validation', 'rgb'), get_loader('validation', 'flow'), get_loader('validation', 'obj')]
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores_early_recognition_fusion(model, loaders)
        else:
            epoch, perf, _ = load_checkpoint(model, best=True)
            print(
                f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")

            loader = get_loader('validation')

            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels, ids = get_scores(model, loader, include_discarded=args.ek100)

            if args.ensamble:
                #Second model
                args.slowfastfusion = False
                args.S_ant = 4
                args.S_enc = 6
                args.alpha = 0.5
                model2 = RULSTM(args.num_class, args.feat_in, args.hidden,
                           args.dropout, sequence_completion=args.sequence_completion)
                model2.to(device)
                chk = torch.load('models/ek55_0.5/RULSTM-anticipation_0.5_6_4_rgb_best.pth.tar')
                epoch2 = chk['epoch']
                best_perf2 = chk['best_perf']
                perf2 = chk['perf']
                model2.load_state_dict(chk['state_dict'])
                print(
                    f"Loaded checkpoint for model2 {type(model2)}. Epoch: {epoch2}. Perf: {perf2:0.2f}.")
                loader2 = get_loader('validation', split_point=1.0)
                verb_scores2, noun_scores2, action_scores2, verb_labels2, noun_labels2, action_labels2, ids2 = get_scores(model2, loader2, include_discarded=args.ek100)

                #third model
                args.S_ant = 16
                args.S_enc = 12
                args.alpha = 0.125
                model1 = RULSTM(args.num_class, args.feat_in, args.hidden,
                           args.dropout, sequence_completion=args.sequence_completion)
                model1.to(device)
                chk = torch.load('models/ek55_0.125/RULSTM-anticipation_0.125_12_16_rgb_best.pth.tar')
                epoch1 = chk['epoch']
                best_perf1 = chk['best_perf']
                perf1 = chk['perf']
                model1.load_state_dict(chk['state_dict'])

                print(
                    f"Loaded checkpoint for model1 {type(model1)}. Epoch: {epoch1}. Perf: {perf1:0.2f}.")
                loader1 = get_loader('validation', split_point=1.0)
                verb_scores1, noun_scores1, action_scores1, verb_labels1, noun_labels1, action_labels1, ids1 = get_scores(model1, loader1, include_discarded=args.ek100)

                args.S_ant = 4
                args.S_enc = 6
                args.alpha = 0.5


                #print(ids.shape)
                #alighnment and average
                #print(verb_scores1.shape, noun_scores1.shape, action_scores1.shape, verb_labels1.shape, noun_labels1.shape, action_labels1.shape, ids1.shape)
                #print(verb_scores2.shape, noun_scores2.shape, action_scores2.shape, verb_labels2.shape, noun_labels2.shape, action_labels2.shape, ids2.shape)
                """verb_scores = np.zeros([verb_scores0.shape[0], verb_scores1.shape[1], verb_scores1.shape[2]])
                noun_scores = np.zeros([noun_scores2.shape[0], noun_scores1.shape[1], noun_scores1.shape[2]])
                action_scores = np.zeros([action_scores2.shape[0], action_scores1.shape[1], action_scores1.shape[2]])
                verb_labels = np.zeros(verb_labels0.shape)
                noun_labels = np.zeros(noun_labels0.shape)
                action_labels = np.zeros(action_labels0.shape)
                ids = np.zeros(ids0.shape)"""
                counter1 = 0
                counter2 = 0
                for i in range(ids.shape[0]):
                    if(ids[i] in ids1):
                        idx = np.where(ids1==ids[i])
                        action_scores[i] = action_scores1[idx, np.arange(0,action_scores1.shape[1],4)] #, action_scores1[i, np.arange(0,action_scores1.shape[1],4)]], axis=0)
                        verb_scores[i] = verb_scores1[idx, np.arange(0,verb_scores1.shape[1],4)] #, verb_scores1[i, np.arange(0,action_scores1.shape[1],4)]], axis=0)
                        noun_scores[i] = noun_scores1[idx, np.arange(0,noun_scores1.shape[1],4)] #, noun_scores1[i, np.arange(0,action_scores1.shape[1],4)]], axis=0)
                        counter1 += 1
                    if(ids[i] in ids2):
                        idx = np.where(ids2==ids[i])
                        action_scores[i] = np.add(0.55*action_scores2[idx], 0.45*action_scores[i]) #, axis=0) 
                        verb_scores[i] = np.add(0.55*verb_scores2[idx],  0.45*verb_scores[i]) #, axis=0)
                        noun_scores[i] = np.add(0.55*noun_scores2[idx], 0.45*noun_scores[i]) #, axis=0)
                        counter2 += 1
                print(counter1, counter2, counter1+counter2)
            #print(verb_scores.shape, noun_scores.shape, action_scores.shape, verb_labels.shape, noun_labels.shape, action_labels.shape, ids.shape)


        if not args.ek100:
            #print(action_scores.shape, action_labels.shape)
            verb_accuracies = topk_accuracy_multiple_timesteps(
                verb_scores, verb_labels)
            noun_accuracies = topk_accuracy_multiple_timesteps(
                noun_scores, noun_labels)
            action_accuracies = topk_accuracy_multiple_timesteps(
                action_scores, action_labels)

            many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot()

            verb_recalls = topk_recall_multiple_timesteps(
                verb_scores, verb_labels, k=5, classes=many_shot_verbs)
            noun_recalls = topk_recall_multiple_timesteps(
                noun_scores, noun_labels, k=5, classes=many_shot_nouns)
            action_recalls = topk_recall_multiple_timesteps(
                action_scores, action_labels, k=5, classes=many_shot_actions)

            all_accuracies = np.concatenate(
                [verb_accuracies, noun_accuracies, action_accuracies, verb_recalls, noun_recalls, action_recalls])
            all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
            indices = [
                ('Verb', 'Top-1 Accuracy'),
                ('Verb', 'Top-5 Accuracy'),
                ('Verb', 'Mean Top-5 Recall'),
                ('Noun', 'Top-1 Accuracy'),
                ('Noun', 'Top-5 Accuracy'),
                ('Noun', 'Mean Top-5 Recall'),
                ('Action', 'Top-1 Accuracy'),
                ('Action', 'Top-5 Accuracy'),
                ('Action', 'Mean Top-5 Recall'),
            ]
            if args.task == 'anticipation':
                cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
            else:
                cc = [f"{c:0.1f}%" for c in np.linspace(0,100,args.S_ant+1)[1:]]

            scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))

            """acts = [474, 917, 1244, 1352, 1542, 1545, 1913, 1923, 1931, 2125, 2177, 2247, 2279, 2300]
            counter = [0 for x in range(len(acts))]
            for i in action_labels:
                for x in range(len(acts)):
                    if acts[x] == i:
                        counter[x] +=1
            print(len(action_labels), counter)"""
            """#per class accuracy
            all_classes_acc = []
            classes = []
            classes_file = open("classes_fused_0.5_ext", 'w')
            scores_file = open("scores_fused_0.5_ext", 'w')
            indices_action = [
                ('Action', 'Top-1 Accuracy'),
                ('Action', 'Top-5 Accuracy'),
            ]
            counter = 0 
            for i in range(action_scores.shape[-1]):
                action_accuracies_per_class = topk_accuracy_multiple_timesteps(action_scores, action_labels, selected_class=i)
                #print(action_accuracies_per_class, action_accuracies_per_class.shape)
                #break
                if(action_accuracies_per_class[1,-1] != 0 and i in action_labels):
                    all_classes_acc.append(action_accuracies_per_class[1,-1])
                    classes.append(i)
                    classes_file.write(str(i)+"\n")
                    scores_file.write(str(action_accuracies_per_class[1,-1])+"\n")
                    counter += 1

                scores_per_class = pd.DataFrame(action_accuracies_per_class*100, columns=cc, index=pd.MultiIndex.from_tuples(indices_action))
                #print(action_scores.shape[-1], scores_per_class)
            print(counter)
            classes_file.close()
            scores_file.close()
            #print(all_classes_acc.sort())"""
        else:
            verb_accuracies = topk_accuracy_multiple_timesteps(
                verb_scores, verb_labels)
            noun_accuracies = topk_accuracy_multiple_timesteps(
                noun_scores, noun_labels)
            action_accuracies = topk_accuracy_multiple_timesteps(
                action_scores, action_labels)

            overall_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores, verb_labels, k=5)
            overall_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores, noun_labels, k=5)
            overall_action_recalls = topk_recall_multiple_timesteps(
                action_scores, action_labels, k=5)

            unseen, tail_verbs, tail_nouns, tail_actions = get_validation_ids()

            unseen_bool_idx = pd.Series(ids).isin(unseen).values
            tail_verbs_bool_idx = pd.Series(ids).isin(tail_verbs).values
            tail_nouns_bool_idx = pd.Series(ids).isin(tail_nouns).values
            tail_actions_bool_idx = pd.Series(ids).isin(tail_actions).values

            tail_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores[tail_verbs_bool_idx], verb_labels[tail_verbs_bool_idx], k=5)
            tail_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores[tail_nouns_bool_idx], noun_labels[tail_nouns_bool_idx], k=5)
            tail_action_recalls = topk_recall_multiple_timesteps(
                action_scores[tail_actions_bool_idx], action_labels[tail_actions_bool_idx], k=5)


            unseen_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores[unseen_bool_idx], verb_labels[unseen_bool_idx], k=5)
            unseen_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores[unseen_bool_idx], noun_labels[unseen_bool_idx], k=5)
            unseen_action_recalls = topk_recall_multiple_timesteps(
                action_scores[unseen_bool_idx], action_labels[unseen_bool_idx], k=5)

            all_accuracies = np.concatenate(
                [verb_accuracies, noun_accuracies, action_accuracies, overall_verb_recalls, overall_noun_recalls, overall_action_recalls, unseen_verb_recalls, unseen_noun_recalls, unseen_action_recalls, tail_verb_recalls, tail_noun_recalls, tail_action_recalls]
            ) #9 x 8

            #all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
            indices = [
                ('Verb', 'Top-1 Accuracy'),
                ('Verb', 'Top-5 Accuracy'),
                ('noun', 'Top-1 Accuracy'),
                ('noun', 'Top-5 Accuracy'),
                ('action', 'Top-1 Accuracy'),
                ('action', 'Top-5 Accuracy'),
                ('Overall Mean Top-5 Recall', 'Verb'),
                ('Overall Mean Top-5 Recall', 'Noun'),
                ('Overall Mean Top-5 Recall', 'Action'),
                ('Unseen Mean Top-5 Recall', 'Verb'),
                ('Unseen Mean Top-5 Recall', 'Noun'),
                ('Unseen Mean Top-5 Recall', 'Action'),
                ('Tail Mean Top-5 Recall', 'Verb'),
                ('Tail Mean Top-5 Recall', 'Noun'),
                ('Tail Mean Top-5 Recall', 'Action'),
            ]

            if args.task == 'anticipation':
                cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
            else:
                cc = [f"{c:0.1f}%" for c in np.linspace(0,100,args.S_ant+1)[1:]]

            scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))

        print(scores)

        if args.task == 'anticipation':
            cc = np.linspace(args.alpha*args.S_ant, 0, args.S_ant+1, dtype=float)
            tta_verb = tta(verb_scores, verb_labels, cc)
            tta_noun = tta(noun_scores, noun_labels, cc)
            tta_action = tta(action_scores, action_labels, cc)

            print(
                f"\nMean TtA(5): VERB: {tta_verb:0.2f} NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}")

    elif args.mode == 'validate':
        if args.task == 'early_recognition' and args.modality == 'fusion':
            loaders = [get_loader('validation', 'rgb'), get_loader('validation', 'flow'),
                       get_loader('validation', 'obj')]
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores_early_recognition_fusion(
                model, loaders)
        else:
            epoch, perf, _ = load_checkpoint(model, best=True)
            print(
                f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")

            loader = get_loader('validation')

            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels,_ = get_scores(model,
                                                                                                              loader)
    elif 'test' in args.mode:
        if args.ek100:
            mm = ['timestamps']
        else:
            mm = ['seen', 'unseen']
        for m in mm:
            if args.task == 'early_recognition' and args.modality == 'fusion':
                loaders = [get_loader(f"test_{m}", 'rgb'), get_loader(f"test_{m}", 'flow'), get_loader(f"test_{m}", 'obj')]
                discarded_ids = loaders[0].dataset.discarded_ids
                verb_scores, noun_scores, action_scores, ids = get_scores_early_recognition_fusion(model, loaders)
            else:
                loader = get_loader(f"test_{m}")
                epoch, perf, _ = load_checkpoint(model, best=True)

                discarded_ids = loader.dataset.discarded_ids

                print(
                    f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")

                verb_scores, noun_scores, action_scores, ids = get_scores(model, loader)

            idx = -4 if args.task == 'anticipation' else -1
            ids = list(ids) + list(discarded_ids)
            verb_scores = np.concatenate((verb_scores, np.zeros((len(discarded_ids), *verb_scores.shape[1:])))) [:,idx,:]
            noun_scores = np.concatenate((noun_scores, np.zeros((len(discarded_ids), *noun_scores.shape[1:])))) [:,idx,:]
            action_scores = np.concatenate((action_scores, np.zeros((len(discarded_ids), *action_scores.shape[1:])))) [:,idx,:]

            actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
            # map actions to (verb, noun) pairs
            a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
                       for a in actions.iterrows()}

            preds = predictions_to_json(verb_scores, noun_scores, action_scores, ids, a_to_vn, version = '0.2' if args.ek100 else '0.1', sls=True)

            if args.ek100:
                with open(join(args.json_directory,exp_name+f"_test.json"), 'w') as f:
                    f.write(json.dumps(preds, indent=4, separators=(',',': ')))
            else:
                with open(join(args.json_directory,exp_name+f"_{m}.json"), 'w') as f:
                    f.write(json.dumps(preds, indent=4, separators=(',',': ')))
    elif 'validate_json' in args.mode:
        if args.task == 'early_recognition' and args.modality == 'fusion':
            loaders = [get_loader("validation", 'rgb'), get_loader("validation", 'flow'), get_loader("validation", 'obj')]
            discarded_ids = loaders[0].dataset.discarded_ids
            verb_scores, noun_scores, action_scores, ids = get_scores_early_recognition_fusion(model, loaders)
        else:
            loader = get_loader("validation")
            epoch, perf, _ = load_checkpoint(model, best=True)

            discarded_ids = loader.dataset.discarded_ids

            print(
                f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")

            verb_scores, noun_scores, action_scores, ids = get_scores(model, loader, challenge=True)

        idx = -4 if args.task == 'anticipation' else -1
        ids = list(ids) + list(discarded_ids)
        verb_scores = np.concatenate((verb_scores, np.zeros((len(discarded_ids), *verb_scores.shape[1:])))) [:,idx,:]
        noun_scores = np.concatenate((noun_scores, np.zeros((len(discarded_ids), *noun_scores.shape[1:])))) [:,idx,:]
        action_scores = np.concatenate((action_scores, np.zeros((len(discarded_ids), *action_scores.shape[1:])))) [:,idx,:]

        actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
        # map actions to (verb, noun) pairs
        a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
                   for a in actions.iterrows()}

        preds = predictions_to_json(verb_scores, noun_scores, action_scores, ids, a_to_vn, version = '0.2' if args.ek100 else '0.1', sls=True)

        with open(join(args.json_directory,exp_name+f"_validation.json"), 'w') as f:
            f.write(json.dumps(preds, indent=4, separators=(',',': ')))

if __name__ == '__main__':
    main()
