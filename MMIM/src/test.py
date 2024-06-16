import torch
import argparse
import numpy as np

from cmams import CMAM
from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader
from model import MMIM

from sklearn.metrics import classification_report, accuracy_score, f1_score

if __name__ == "__main__":
    args = get_args()
    dataset = str.lower(args.dataset.strip())

    print("Start loading the data....")
    train_config = get_config(dataset, mode="train", batch_size=args.batch_size)
    valid_config = get_config(dataset, mode="valid", batch_size=args.batch_size)
    test_config = get_config(dataset, mode="test", batch_size=args.batch_size)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(args, train_config, shuffle=True)
    print("Training data loaded!")
    valid_loader = get_loader(args, valid_config, shuffle=False)
    print("Validation data loaded!")
    test_loader = get_loader(args, test_config, shuffle=False)
    print("Test data loaded!")
    print("Finish loading the data....")

    torch.autograd.set_detect_anomaly(True)

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, "MSELoss")

    # model
    if args.train_method == "missing":
        save_mode = f"0"
    elif args.train_method == "g_noise":
        save_mode = f"N"
    elif args.train_method == "hybird":
        save_mode = f"H"
    else:
        raise
    model = MMIM(args)

    if args.dataset == "mosi":
        model = torch.load(args.save_model_to)

        model = model.cuda()
    elif args.dataset == "mosei":
        model = torch.load(args.save_model_to)

    if args.cmam_path is not None:
        if args.dataset == "mosi":
            cmam = CMAM(
                args,
                acoustic_enc_state_dict=args.audio_encoder_path,
                visual_enc_state_dict=args.video_encoder_path,
            )
            cmam.load_state_dict(torch.load(args.cmam_path))
            cmam = cmam.cuda()
            model.cmam = cmam
            model.cmam = model.cmam.eval()
            print("Loading CMAM model from: ", args.cmam_path)

    solver = Solver(
        args,
        train_loader=train_loader,
        dev_loader=valid_loader,
        test_loader=test_loader,
        is_train=False,
        model=model,
    )

    test_loss, results, truths = solver.evaluate(
        solver.model, solver.criterion, test=True
    )

    if args.dataset == "mosi":
        # eval_mosi(results, truths, True)
        ##########################################################
        y_pred = results
        y_true = truths
        accuracy = calc_metrics(
            y_true, y_pred, "test", True, save_path=args.save_results_to
        )
        print("------------------------------------------------------")

        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        binary_truth = y_true[non_zeros] > 0
        binary_preds = y_pred[non_zeros] > 0
        acc2 = accuracy_score(binary_truth.cpu(), binary_preds.cpu())
        print("acc2:", acc2)
        # print(torch.eq(binary_truth, binary_preds))
        idx_diff = []
        for i in range(len(binary_preds)):
            if binary_preds[i][0] == binary_truth[i][0]:
                pass
            else:
                idx_diff.append(i)
        # print("idx_diff:", idx_diff)
        ##########################################################
    elif args.dataset == "mosei":
        # eval_mosei_senti(results, truths, True)
        ##########################################################
        y_pred = results
        y_true = truths
        accuracy = calc_metrics(
            y_true, y_pred, "test", True, save_path=args.save_results_to
        )
        ##########################################################
        print("------------------------------------------------------")
        print("y_pred:", y_pred.shape)
        print("y_true:", y_true.shape)

        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        binary_truth = y_true[non_zeros] > 0
        binary_preds = y_pred[non_zeros] > 0
        acc2 = accuracy_score(binary_truth.cpu(), binary_preds.cpu())
        print("acc2:", acc2)
        # print(torch.eq(binary_truth, binary_preds))
        idx_diff = []
        for i in range(len(binary_preds)):
            if binary_preds[i][0] == binary_truth[i][0]:
                pass
            else:
                idx_diff.append(i)
        print("idx_diff:", idx_diff)
    # solver.train_and_eval()
