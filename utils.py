import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


def softmax_score(score1, score2):
    score = torch.transpose(torch.stack((score1, score2)), 0, 1)
    score = F.softmax(score, dim=-1)
    return score[:, 0], score[:, 1]


def get_res(score1, score2, threshold=0.2):
    score = torch.transpose(torch.stack((score1, score2)), 0, 1)
    score_abs = torch.abs(score1 - score2)
    mask = score_abs < threshold
    res = torch.zeros_like(score)
    res[torch.arange(score.size(0)), torch.argmax(score, dim=1)] = 1
    res[mask] = 0.5
    return res


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def eval_HPD(res_path):
    df = pd.read_csv(res_path)
    d = {}
    print(df.shape[0])
    prompt = ''
    total = 0
    cor = 0
    for i in range(df.shape[0]):
        if df.iloc[i]['prompt'] != prompt:
            prompt = df.iloc[i]['prompt']
            d[prompt] = []
            total += 1
        d[prompt].append(df.iloc[i]['score_0'] > df.iloc[i]['score_1'])
    for i in d.keys():
        if False not in d[i]:
            cor += 1
    print(cor, '/', total, ' = ', cor / total)


def SROCC(true_rank, pred_rank):
    if sum(pred_rank) == len(pred_rank):
        pred_rank = np.linspace(1, len(pred_rank), len(pred_rank))
    n = len(true_rank)
    s = sum((pred_rank - true_rank) ** 2)
    return 1 - 6 * s / (n * (n ** 2 - 1))


def KROCC(true_rank, pred_rank):
    if sum(pred_rank) == len(pred_rank):
        pred_rank = np.linspace(1, len(pred_rank), len(pred_rank))
    n = len(true_rank)
    p = q = x = y = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (true_rank[i] - true_rank[j]) * (pred_rank[i] - pred_rank[j]) > 0:
                p += 1
            elif true_rank[i] == true_rank[j] and pred_rank[i] != pred_rank[j]:
                x += 1
            elif true_rank[i] != true_rank[j] and pred_rank[i] == pred_rank[j]:
                y += 1
            elif true_rank[i] != true_rank[j] and pred_rank[i] != pred_rank[j]:
                q += 1
    return (p - q) / np.sqrt((p + q + x) * (p + q + y))


def PLCC(true_rank, pred_rank):
    if sum(pred_rank) == len(pred_rank):
        pred_rank = np.linspace(1, len(pred_rank), len(pred_rank))
    true_m = np.mean(true_rank)
    pred_m = np.mean(pred_rank)
    return sum((true_rank - true_m) * (pred_rank - pred_m)) / np.sqrt(sum((true_rank - true_m) ** 2) * sum((pred_rank - pred_m) ** 2))