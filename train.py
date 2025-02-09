import clip
import warnings
import sys
import random
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from data_loader import get_dataloader
from model import Projection, Net, PerceptualNet, TextEncoder, ImageEncoder, Net_blip, TextEncoder_blip, ImageEncoder_blip, TextEncoder_blip1, ImageEncoder_blip1
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from lavis.models import load_model
from transformers import BertTokenizer
from utils import get_res, get_model_info, softmax_score, convert_models_to_fp32
from loss import InfoNCE, CrossEntropy


sys.setrecursionlimit(100000)
warnings.filterwarnings('ignore')


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


setup_seed(2023)
"""
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py --id 1110 --is_train --is_eval --epochs 1 --batch 128 --nw 0 --dataset Pick-a-pic --lr1 1e-6 --lr2 5e-5 --tau 0.1 --is_save --check steps --model_type blip --layers > p1 2>&1 &
"""
parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--id', type=int, default=random.randint(0, 100000))
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--is_eval', action='store_true')
parser.add_argument('--is_save', action='store_true')
parser.add_argument('--layers', action='store_true')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--nw', type=int, default=0)
parser.add_argument('--dataset', type=str, default='Pick-a-pic')  # '1'
parser.add_argument('--model_type', type=str, default='clip')  # 'blip', 'blip1'
parser.add_argument('--lr1', type=float, default=3e-6)
parser.add_argument('--lr2', type=float, default=3e-6)
parser.add_argument('--tau', type=float, default=0.04)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--lambda2', type=float, default=0.5)
parser.add_argument('--check', type=str, default='epochs')  # 'steps'
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
enable_amp = False if device == "cpu" else True

tid = args.id
is_train = args.is_train
is_eval = args.is_eval
is_save = args.is_save
data_set = args.dataset
learning_rate1 = args.lr1
learning_rate2 = args.lr2
batch_size = args.batch
num_workers = args.nw
epochs = args.epochs
tau = args.tau
lambda1 = args.lambda1
lambda2 = args.lambda2
model_type = args.model_type
layers = args.layers
check = args.check
threshold = 0
print('id:', tid, 'is_train:', is_train, 'is_eval:', is_eval, 'is_save:', is_save, 'dataset:', data_set,
      'lr1:', learning_rate1, 'lr2:', learning_rate2, 'epochs:', epochs, 'batch_size:', batch_size,
      'num_workers:', num_workers, 'tau:', tau, 'lambda1:', lambda1, 'lambda2:', lambda2,
      'model_type:', model_type, 'layers:', layers, 'check:', check)

if data_set == 'Pick-a-pic':
    train_data_root = r'../data/PaP/train'
    val_data_root = r'../data/PaP/test'
elif data_set == 'ImageR':
    train_data_root = '../data/IR/train/'
    val_data_root = '../data/IR/test/'
elif data_set == 'HPDv1':
    train_data_root = val_data_root = '../data/HPSv1/'
else:
    # train_data_root = '../data/Project_Dataset_1/Selected_Train_Dataset/'
    # val_data_root = '../data/Project_Dataset_1/Selected_Train_Dataset/'
    train_data_root = '../data/Project_Dataset_train_val/train/'
    val_data_root = '../data/Project_Dataset_train_val/val/'

train_dataloader = get_dataloader(train_data_root, batch_size, data_set, num_workers, 'train', model_type)
val_dataloader = get_dataloader(val_data_root, batch_size, data_set, num_workers, 'val', model_type)
print(train_dataloader.dataset.__len__())
print(val_dataloader.dataset.__len__())


def get_models(device, model_type, layers):
    if model_type == 'clip':
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        if device == "cpu":
            model.float()
        else:
            clip.model.convert_weights(model)
        proj = Projection().to(device)  # 用于InfoNCE计算前的embedding的投影函数
        textencoder = TextEncoder(model).to(device)
        imageencoder = ImageEncoder(model).to(device)
        net = Net(imageencoder, textencoder, proj).to(device)  # 从图片、文本到它们的embedding的网络
    elif model_type == 'blip':
        model = load_model("blip2_image_text_matching", "pretrain")
        textencoder = TextEncoder_blip(
            bert=model.Qformer.bert,
            text_proj=model.text_proj).to(device)
        imageencoder = ImageEncoder_blip(
            visual_encoder=model.visual_encoder,
            ln_vision=model.ln_vision,
            query_tokens=model.query_tokens,
            bert=model.Qformer.bert,
            vision_proj=model.vision_proj,
            layers=layers).to(device)
        net = Net_blip(imageencoder, textencoder).to(device)  # 从图片、文本到它们的embedding的网络
    else:
        model = load_model("blip_image_text_matching", "base")
        textencoder = TextEncoder_blip1(
            text_encoder=model.text_encoder,
            text_proj=model.text_proj
        ).to(device)
        imageencoder = ImageEncoder_blip1(
            visual_encoder=model.visual_encoder,
            vision_proj=model.vision_proj,
            layers=layers).to(device)
        net = Net_blip(imageencoder, textencoder).to(device)  # 从图片、文本到它们的embedding的网络

    p_net = PerceptualNet(mode=model_type).to(device)  # 感知模块

    if torch.cuda.device_count() > 1:
        textencoder = DataParallel(textencoder)
        imageencoder = DataParallel(imageencoder)
        net = DataParallel(net)
        p_net = DataParallel(p_net)
    return model, net, p_net


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})
model, net, p_net = get_models(device, model_type, layers)
get_model_info(net)
get_model_info(p_net)
optimizer1 = optim.AdamW(net.parameters(), lr=learning_rate1, weight_decay=0.2)
optimizer2 = optim.AdamW(p_net.parameters(), lr=learning_rate2, weight_decay=0.2)
if check == 'steps':
    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=600, gamma=0.2)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=600, gamma=0.5)
else:
    scheduler1 = lr_scheduler.ExponentialLR(optimizer1, gamma=0.9)#0.9
    scheduler2 = lr_scheduler.ExponentialLR(optimizer2, gamma=0.1)#0.9

scaler = GradScaler(enabled=enable_amp)  # 混合精度训练


def eval(check_type, steps):
    with torch.no_grad():
        epoch_loss_v = 0
        e_loss1_v = 0
        e_loss2_v = 0
        cor1 = 0
        cor2 = 0
        cor = 0
        score_total = [[], []]
        score_a = [[], []]
        score_p = [[], []]
        prompt_list = []
        net.eval()
        p_net.eval()
        for batch_e, (image1, image2, prompt, label) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)

            if model_type == 'clip':
                prompt_tokens = clip.tokenize(prompt, truncate=True).to(device)
                image1_embedding, image2_embedding, prompts_embedding = net(image1, image2, prompt_tokens)
            else:
                prompt_tokens = tokenizer(prompt, truncation=True, padding='max_length', max_length=77, return_tensors="pt", ).to(device)
                input_ids = prompt_tokens.input_ids.to(device)
                attention_mask = prompt_tokens.attention_mask.to(device)
                image1_embedding, image2_embedding, prompts_embedding = net(image1, image2, input_ids, attention_mask)
            out = p_net(image1, image2)

            loss1 = InfoNCE(image1_embedding, image2_embedding, prompts_embedding, label, tau=tau)
            loss2 = CrossEntropy(out, label)
            loss = loss1 + loss2

            e_loss1_v += loss1.item()
            e_loss2_v += loss2.item()
            epoch_loss_v += loss.item()

            score_a1 = F.cosine_similarity(image1_embedding, prompts_embedding)
            score_a2 = F.cosine_similarity(image2_embedding, prompts_embedding)
            score_a1, score_a2 = softmax_score(score_a1, score_a2)
            score_b1 = F.softmax(out, dim=-1)[:, 0]
            score_b2 = F.softmax(out, dim=-1)[:, 1]
            labels_0 = label[:, 0] == 1
            score_1 = torch.where(labels_0, lambda1 * score_a1 + lambda2 * score_b1,
                                  lambda1 * score_a2 + lambda2 * score_b2)
            score_2 = torch.where(labels_0, lambda1 * score_a2 + lambda2 * score_b2,
                                  lambda1 * score_a1 + lambda2 * score_b1)

            res_b = get_res(score_b1, score_b2, threshold)
            res_a = get_res(score_a1, score_a2, threshold)

            prompt_list.extend(prompt)
            score_a[0].extend(torch.where(labels_0, score_a1, score_a2).tolist())
            score_a[1].extend(torch.where(labels_0, score_a2, score_a1).tolist())
            score_p[0].extend(torch.where(labels_0, score_b1, score_b2).tolist())
            score_p[1].extend(torch.where(labels_0, score_b2, score_b1).tolist())
            score_total[0].extend(score_1.tolist())
            score_total[1].extend(score_2.tolist())

            cor1 += (torch.sum(res_a == label, dim=1) > 0).sum().item()
            cor2 += (torch.sum(res_b == label, dim=1) > 0).sum().item()
            cor += (score_1 > score_2).sum().item()

        print(check_type, '%d, loss: %.3f, loss1: %.3f loss2: %.3f' % (steps, epoch_loss_v, e_loss1_v, e_loss2_v))
        print(cor1, '/', val_dataloader.dataset.__len__(), ' = ', cor1 / val_dataloader.dataset.__len__())
        print(cor2, '/', val_dataloader.dataset.__len__(), ' = ', cor2 / val_dataloader.dataset.__len__())
        print(cor, '/', val_dataloader.dataset.__len__(), ' = ', cor / val_dataloader.dataset.__len__())

    if is_save is True:
        data_save = {'prompt': prompt_list, 'scorea_0': score_a[0], 'scorea_1': score_a[1],
                     'scorep_0': score_p[0], 'scorep_1': score_p[1], 'score_0': score_total[0],
                     'score_1': score_total[1]}
        dataframe = pd.DataFrame(data_save)
        dataframe = dataframe.sort_values(by=['prompt'], ascending=True)
        dataframe.to_csv('../res/result%d-%d.csv' % (tid, steps), encoding='utf-8')
        if cor1 / val_dataloader.dataset.__len__() >= 0.99:
            torch.save(net, '../model/net%d-%d.pkl' % (tid, steps))
        # torch.save(model, '../model/model%d-%s.pkl' % (tid, i + 1))  # if not want to save model please comment
        # torch.save(proj, '../model/proj%d-%s.pkl' % (tid, i + 1))  # if not want to save model please comment
        if cor2 / val_dataloader.dataset.__len__() >= 0.99:
            torch.save(p_net, '../model/p_net%d-%d.pkl' % (tid, steps))  # if not want to save model please comment


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    for i in range(epochs):
        epoch_loss = 0
        e_loss1 = 0
        e_loss2 = 0
        """训练过程"""
        if is_train is True:
            for batch, (image1, image2, prompt, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                net.train()
                p_net.train()
                with autocast(enabled=enable_amp):
                    image1 = image1.to(device)
                    image2 = image2.to(device)
                    label = label.to(device)

                    if model_type == 'clip':
                        prompt_tokens = clip.tokenize(prompt, truncate=True).to(device)
                        image1_embedding, image2_embedding, prompts_embedding = net(image1, image2, prompt_tokens)
                    else:
                        prompt_token = tokenizer(prompt, truncation=True, padding='max_length', max_length=77, return_tensors="pt", ).to(device)
                        input_ids = prompt_token.input_ids.to(device)
                        attention_mask = prompt_token.attention_mask.to(device)
                        image1_embedding, image2_embedding, prompts_embedding = net(image1, image2, input_ids, attention_mask)

                    out = p_net(image1, image2)

                    loss1 = InfoNCE(image1_embedding, image2_embedding, prompts_embedding, label, tau=tau)
                    loss2 = CrossEntropy(out, label)
                    loss = loss1 + loss2

                e_loss1 += loss1.item()
                e_loss2 += loss2.item()
                epoch_loss += loss.item()

                optimizer1.zero_grad()
                scaler.scale(loss1).backward()
                optimizer2.zero_grad()
                scaler.scale(loss2).backward()

                if device == "cpu":
                    scaler.step(optimizer1)
                    scaler.step(optimizer2)
                else:
                    if model_type == 'clip':
                        convert_models_to_fp32(model)
                    scaler.step(optimizer1)
                    scaler.step(optimizer2)
                    if model_type == 'clip':
                        clip.model.convert_weights(model)
                scaler.update()

                if check == 'steps':
                    scheduler1.step()
                    scheduler2.step()

                if batch % 100 == 0 and check == 'steps':
                     if is_eval is True:
                        eval('setp', batch)

            print('epoch %d, loss: %.3f, loss1: %.3f loss2: %.3f' % (i + 1, epoch_loss, e_loss1, e_loss2))

            scheduler1.step()
            scheduler2.step()

        if is_eval is True:
            eval('epoch', i + 1)
