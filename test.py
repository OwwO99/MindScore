import clip
import warnings
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from data_loader import get_dataloader
from transformers import BertTokenizer
from utils import get_res, softmax_score


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--nw', type=int, default=4)
parser.add_argument('--dataset', type=str, default='Pick-a-pic')
parser.add_argument('--model_type', type=str, default='clip')  # 'blip'
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--lambda2', type=float, default=0.5)
parser.add_argument('--lambda3', type=float, default=0.5)
parser.add_argument('--model_id', nargs='+', required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

net_path = '../model/net%s-%s.pkl' % (args.model_id[0], args.model_id[1])
p_net_path = '../model/p_net%s-%s.pkl' % (args.model_id[2], args.model_id[3])
r_net_path = '../model/model_iters_%s-%s.pth' % (args.model_id[4], args.model_id[5])
batch_size = args.batch
lambda1 = args.lambda1
lambda2 = args.lambda2
lambda3 = args.lambda3
num_workers = args.nw
model_type = args.model_type
data_set = args.dataset  # 'Pick-a-pic' or others
threshold = 0
print(net_path, p_net_path, 'dataset:', data_set, 'batch_size:', batch_size, 'num_workers:', num_workers,
      'lambda1:', lambda1, 'lambda2:', lambda2, 'model_type:', model_type)

if data_set == 'Pick-a-pic':
    test_data_root = r'../data/PaP/test'
elif data_set == 'ImageR':
    test_data_root = '../data/ImageR/test_rebuild/'
elif data_set == 'HPDv1':
    test_data_root = '../data/HPSv1/'
else:
    test_data_root = '../data/Project_Dataset/Selected_Train_Dataset/'

test_dataloader = get_dataloader(test_data_root, batch_size, data_set, num_workers, 'val', model_type)
print(test_dataloader.dataset.__len__())

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})
net = torch.load(net_path).to(device)  # load已经存好的网络
p_net = torch.load(p_net_path).to(device)
r_net = torch.load(r_net_path).to(device)
net.eval()
p_net.eval()
r_net.eval()

with torch.no_grad():
    cor1 = 0
    cor2 = 0
    cor3 = 0
    cor = 0
    score_total = [[], []]
    score_a = [[], []]
    score_p = [[], []]
    prompt_list = []
    for batch, (image1, image2, prompt, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
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
        score_c1 = r_net(image1).squeeze(1)
        score_c2 = r_net(image2).squeeze(1)
        score_c2, score_c1 = softmax_score(score_c1, score_c2)
        score_a1 = F.cosine_similarity(image1_embedding, prompts_embedding)
        score_a2 = F.cosine_similarity(image2_embedding, prompts_embedding)
        score_a1, score_a2 = softmax_score(score_a1, score_a2)
        score_b1 = F.softmax(out, dim=-1)[:, 0]
        score_b2 = F.softmax(out, dim=-1)[:, 1]
        # score_1 = lambda1 * score_a1 + lambda2 * score_b1
        # score_2 = lambda1 * score_a2 + lambda2 * score_b2
        labels_0 = label[:, 0] == 1
        score_1 = torch.where(labels_0, lambda1 * score_a1 + lambda2 * score_b1 + lambda3 * score_c1, lambda1 * score_a2 + lambda2 * score_b2 + lambda3 * score_c2)
        score_2 = torch.where(labels_0, lambda1 * score_a2 + lambda2 * score_b2 + lambda3 * score_c2, lambda1 * score_a1 + lambda2 * score_b1 + lambda3 * score_c1)

        prompt_list.extend(prompt)
        # score_a[0].extend(torch.where(labels_0, score_a1, score_a2).tolist())
        # score_a[1].extend(torch.where(labels_0, score_a2, score_a1).tolist())
        score_a[0].extend(torch.where(labels_0, score_a1, score_a2).tolist())
        score_a[1].extend(torch.where(labels_0, score_a2, score_a1).tolist())
        score_p[0].extend(torch.where(labels_0, score_b1, score_b2).tolist())
        score_p[1].extend(torch.where(labels_0, score_b2, score_b1).tolist())
        # score_total[0].extend(torch.where(labels_0, score_1, score_2).tolist())
        # score_total[1].extend(torch.where(labels_0, score_2, score_1).tolist())
        score_total[0].extend(score_1.tolist())
        score_total[1].extend(score_2.tolist())

        res_a = get_res(score_a1, score_a2, threshold)
        res_b = get_res(score_b1, score_b2, threshold)
        res_c = get_res(score_c1, score_c2, threshold)

        cor1 += (torch.sum(res_a == label, dim=1) > 0).sum().item()
        cor2 += (torch.sum(res_b == label, dim=1) > 0).sum().item()
        cor3 += (torch.sum(res_c == label, dim=1) > 0).sum().item()
        cor += (score_1 > score_2).sum().item()

    print(cor1, '/', test_dataloader.dataset.__len__(), ' = ', cor1 / test_dataloader.dataset.__len__())
    print(cor2, '/', test_dataloader.dataset.__len__(), ' = ', cor2 / test_dataloader.dataset.__len__())
    print(cor, '/', test_dataloader.dataset.__len__(), ' = ', cor / test_dataloader.dataset.__len__())

    data_save = {'prompt': prompt_list, 'scorea_0': score_a[0], 'scorea_1': score_a[1],
                 'scorep_0': score_p[0], 'scorep_1': score_p[1], 'score_0': score_total[0],
                 'score_1': score_total[1]}
    dataframe = pd.DataFrame(data_save)
    dataframe = dataframe.sort_values(by=['prompt'], ascending=True)
    dataframe.to_csv('../res/test_result%s-%s-%s-%s.csv' % (args.model_id[0], args.model_id[1], args.model_id[2], args.model_id[3]), encoding='utf-8')
