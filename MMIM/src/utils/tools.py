import torch
import os
import io


def save_load_name(args, name=""):
    if args.aligned:
        name = name if len(name) > 0 else "aligned_model"
    elif not args.aligned:
        name = name if len(name) > 0 else "nonaligned_model"

    return name + "_" + args.model


def save_model(args, model):
    # name = save_load_name(args, name)
    name = "best_model"
    if not os.path.exists("pre_trained_models"):
        os.mkdir("pre_trained_models")

    model_name = args.save_model_to
    print(f"Saving model to {model_name}")
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    torch.save(model, model_name)


def load_model(args, name=""):
    # name = save_load_name(args, name)
    model = torch.load(args.save_model_to)
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)

    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)

    return t


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x
