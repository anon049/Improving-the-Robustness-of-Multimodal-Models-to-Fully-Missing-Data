import torch
import argparse
import numpy as np

from cmams import CMAM
from utils import *
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader
from model import MMIM


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

    model = torch.load(args.save_model_to)
    acoustic = model.acoustic_enc
    visual_encoder = model.visual_enc

    acoustic_enc_state_dict = acoustic.state_dict()
    visual_enc_state_dict = visual_encoder.state_dict()

    torch.save(acoustic_enc_state_dict, args.audio_encoder_path)
    torch.save(visual_enc_state_dict, args.video_encoder_path)

    print("Acoustic encoder saved to: ", args.audio_encoder_path)
    print("Visual encoder saved to: ", args.video_encoder_path)
