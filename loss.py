import torch
import torch.nn as nn
import torch.nn.functional as F


loss_fn = nn.CrossEntropyLoss()


def InfoNCE(image1_embeddings, image2_embeddings, anchor_embeddings, labels, tau: float = 0.8):
    labels = labels[:, 0] == 1
    labels = labels.unsqueeze(-1)
    pos_embeddings = torch.where(labels, image1_embeddings, image2_embeddings)
    neg_embeddings = torch.where(labels, image2_embeddings, image1_embeddings)
    pos_pair = torch.exp(F.cosine_similarity(pos_embeddings, anchor_embeddings, dim=-1) / tau)
    neg_pair = torch.exp(F.cosine_similarity(neg_embeddings, anchor_embeddings, dim=-1) / tau)
    return -torch.mean(torch.log(pos_pair / (pos_pair + neg_pair)))


def HingeLoss(image1_embeddings, image2_embeddings, anchor_embeddings, labels, margin: float = 0.2):
    labels = labels[:, 0] == 1
    labels = labels.unsqueeze(-1)
    pos_embeddings = torch.where(labels, image1_embeddings, image2_embeddings)
    neg_embeddings = torch.where(labels, image2_embeddings, image1_embeddings)
    pos_scores = F.cosine_similarity(pos_embeddings, anchor_embeddings, dim=-1)
    neg_scores = F.cosine_similarity(neg_embeddings, anchor_embeddings, dim=-1)
    loss = torch.mean(F.relu(margin - pos_scores + neg_scores))
    return loss


def MarginLoss(output, labels, margin: float = 0.1, reduction='mean'):
    labels = labels[:, 0] == 1
    output = F.softmax(output, dim=-1)
    if reduction == 'sum':
        return torch.sum(F.relu(margin + torch.where(labels, output[:, 1] - output[:, 0], output[:, 0] - output[:, 1])))
    else:
        return torch.mean(F.relu(margin + torch.where(labels, output[:, 1] - output[:, 0], output[:, 0] - output[:, 1])))


def CrossEntropy(output, label, margin: float = 0.1):
    return loss_fn(output, label) + MarginLoss(output, label, margin=margin)
