import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader
from model import MMIM


from cmams import CMAM
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)
    r = {
        "MAE": mae,
        "Corr": corr,
        "mult_acc_7": mult_a7,
        "mult_acc_5": mult_a5,
        "f_score": f_score,
        "accuracy": accuracy_score(binary_truth, binary_preds),
    }
    return r


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def evaluate(model, criterion, hp):
    model.eval()
    loader = test_loader
    total_loss = 0.0
    total_l1_loss = 0.0

    results = []
    truths = []

    with torch.no_grad():
        for batch in loader:
            (
                text,
                vision,
                vlens,
                audio,
                alens,
                y,
                lengths,
                bert_sent,
                bert_sent_type,
                bert_sent_mask,
                ids,
            ) = batch

            with torch.cuda.device(0):
                text, audio, vision, y = (
                    text.cuda(),
                    audio.cuda(),
                    vision.cuda(),
                    y.cuda(),
                )
                lengths = lengths.cuda()
                bert_sent, bert_sent_type, bert_sent_mask = (
                    bert_sent.cuda(),
                    bert_sent_type.cuda(),
                    bert_sent_mask.cuda(),
                )
                # if hp.dataset == "iemocap":
                #     y = y.long()

                # if hp.dataset == "ur_funny":
                #     y = y.squeeze()

            batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)

            #####################################
            is_train = model.training

            model.hp.is_test = True
            model.hp.train_method = hp.train_method
            model.hp.train_changed_modal = hp.train_changed_modal
            model.hp.train_changed_pct = hp.train_changed_pct
            model.hp.test_method = hp.test_method
            model.hp.test_changed_modal = hp.test_changed_modal
            model.hp.test_changed_pct = hp.test_changed_pct
            #####################################

            # we don't need lld and bound anymore
            _, _, preds, _, _ = model(
                is_train,
                text,
                vision,
                audio,
                vlens,
                alens,
                bert_sent,
                bert_sent_type,
                bert_sent_mask,
            )

            total_loss += criterion(preds, y).item() * batch_size

            # Collect the results into ntest if test else hp.n_valid)
            results.append(preds)
            truths.append(y)

    avg_loss = total_loss / hp.n_test

    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = get_args()

    epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    args.batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = str.lower(args.dataset.strip())
    os.makedirs("cmam_logs", exist_ok=True)
    tb_logger = SummaryWriter(log_dir="./cmam_logs")
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

    feature_model = MMIM(args)

    model_path = args.baseline_model_path

    feature_model = torch.load(model_path).to(device)

    feature_model = feature_model.train()
    best_loss = 1e6

    model = CMAM(
        args,
        acoustic_enc_state_dict=args.audio_encoder_path,
        visual_enc_state_dict=args.video_encoder_path,
    )

    model = model.to(device)
    model = model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # feature_optimizer = Adam(feature_model.parameters(), lr=5e-3)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.1)
    regression_criterion = torch.nn.L1Loss()

    best_f1 = 0.0
    for epoch in range(epochs):

        train_batch_loss = 0.0
        train_batch_r_loss = 0.0
        train_batch_c_loss = 0.0

        y_true = []
        y_pred = []

        model = model.train()

        for batch in tqdm(train_loader):
            (
                text,
                video,
                vlens,
                audio,
                alens,
                y,
                lengths,
                bert_sent,
                bert_sent_type,
                bert_sent_mask,
                ids,
            ) = batch
            audio = audio.to(device)
            video = video.to(device)
            text = text.to(device)
            bert_sent = bert_sent.to(device)
            bert_sent_mask = bert_sent_mask.to(device)
            bert_sent_type = bert_sent_type.to(device)
            y = y.to(device)

            feature_model.hp.is_test = False
            with torch.no_grad():
                # we don't need lld and bound anymore
                _, _, preds, _, _, target = feature_model(
                    False,
                    text,
                    video,
                    audio,
                    vlens,
                    alens,
                    bert_sent,
                    bert_sent_type,
                    bert_sent_mask,
                    ret_txt_f=True,
                )

            optimizer.zero_grad()
            # feature_optimizer.zero_grad()
            output = model((audio, alens), (video, vlens))
            loss = regression_criterion(output, target)

            train_batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # feature_optimizer.step()

            y_true.append(target.cpu().detach().numpy())
            y_pred.append(output.cpu().detach().numpy())

        train_batch_loss /= len(train_loader)

        validation_loss = 0.0
        model = model.eval()

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                (
                    text,
                    video,
                    vlens,
                    audio,
                    alens,
                    y,
                    lengths,
                    bert_sent,
                    bert_sent_type,
                    bert_sent_mask,
                    ids,
                ) = batch
                audio = audio.to(device)
                video = video.to(device)
                text = text.to(device)
                bert_sent = bert_sent.to(device)
                bert_sent_mask = bert_sent_mask.to(device)
                bert_sent_type = bert_sent_type.to(device)
                y = y.to(device)
                feature_model.hp.is_test = False

                _, _, preds, _, _, target = feature_model(
                    False,
                    text,
                    video,
                    audio,
                    vlens,
                    alens,
                    bert_sent,
                    bert_sent_type,
                    bert_sent_mask,
                    ret_txt_f=True,
                )
                output = model((audio, alens), (video, vlens))
                loss = regression_criterion(output, target)

                validation_loss += loss.item()
                y_true.append(target.cpu().detach().numpy())
                y_pred.append(output.cpu().detach().numpy())

        validation_loss /= len(valid_loader)
        scheduler.step(validation_loss)

        tb_logger.add_scalars(
            "loss", {"train": train_batch_loss, "validation": validation_loss}, epoch
        )

        ## evaluate the model using the generated features
        feature_model.cmam = model
        test_loss, results, truths = evaluate(
            feature_model, regression_criterion, hp=args
        )
        feature_model.cmam = None

        evaluation_results = eval_mosei_senti(results=results, truths=truths)
        evaluation_table = pd.DataFrame(
            index=[
                "mae",
                "corr",
                "mult_acc_7",
                "mult_acc_5",
                "acc",
                "f1",
            ],
            columns=["results"],
        )
        evaluation_table.loc["mae"] = evaluation_results["MAE"]
        evaluation_table.loc["corr"] = evaluation_results["Corr"]
        evaluation_table.loc["f1"] = evaluation_results["f_score"]
        evaluation_table.loc["mult_acc_7"] = evaluation_results["mult_acc_7"]
        evaluation_table.loc["mult_acc_5"] = evaluation_results["mult_acc_5"]
        evaluation_table.loc["acc"] = evaluation_results["accuracy"]
        os.makedirs(os.path.dirname(args.cmam_path), exist_ok=True)

        if best_loss > validation_loss:
            best_loss = validation_loss
            print("Saving model")
            torch.save(
                model.state_dict(),
                os.path.join(
                    os.path.dirname(args.cmam_path),
                    f"cmam_loss.pt",
                ),
            )
        if best_f1 < evaluation_results["f_score"]:
            best_f1 = evaluation_results["f_score"]
            print("Saving model")
            torch.save(
                model.state_dict(),
                os.path.join(
                    os.path.dirname(args.cmam_path),
                    f"cmam_f1.pt",
                ),
            )
        table = pd.DataFrame(
            index=["train", "validation"],
            columns=["loss"],
        )
        table.loc["train", "loss"] = train_batch_loss
        table.loc["validation", "loss"] = validation_loss

        print(f"Epoch {epoch + 1}/{epochs}")
        print(table.to_markdown(tablefmt="grid"))
        print(evaluation_table.T.to_markdown(tablefmt="grid"))
        print("\n")
        print("-" * 50)
        print(f"BEST LOSS: {best_loss}")
        print(f"BEST F1: {best_f1}")
        print("-" * 50)
    tb_logger.close()
