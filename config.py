from argparse import ArgumentParser
import json

def get_args(stdin):
    parser = ArgumentParser(stdin, description="Training program for RULSTM")
    parser.add_argument('mode', type=str, choices=['train', 'validate', 'test', 'test', 'validate_json'], 
                        default='train', help="Whether to perform training, validation or test.\
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
    parser.add_argument('--task', type=str, default='anticipation', choices=['anticipation', 'early_recognition'], 
                        help='Task to tackle: anticipation or early recognition')
    parser.add_argument('--img_tmpl', type=str, default='frame_{:010d}.jpg', 
                        help='Template to use to load the representation of a given frame')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'flow', 'obj', 'fusion'], 
                        help="Modality. Rgb/flow/obj represent single branches, whereas fusion indicates the whole \
                                model with modality attention.")
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
    parser.add_argument('--ignore_checkpoints', action='store_true',
                        help='If specified, avoid loading existing models (no pre-training)')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume suspended training')
    parser.add_argument('--ek100', action='store_true',
                        help="Whether to use EPIC-KITCHENS-100")
    parser.add_argument('--json_directory', type=str, default=None, 
                        help='Directory in which to save the generated jsons.')
    parser.add_argument('--ensamble', action='store_true')

    # Parse args
    args = parser.parse_args()

    # Process args
    if args.slowfastfusion:
        args.alpha = min(args.alphas_fused)
        args.S_enc = max(args.S_enc_fused)
        args.S_ant = max(args.S_ant_fused)

    print('Input Args:', json.dumps(vars(args), indent=4))
    return args
