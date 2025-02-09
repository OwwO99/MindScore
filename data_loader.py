import io
import os
import re
import glob
import json
import random
import clip
import torch
import pyarrow.parquet as pq
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import InterpolationMode
from lavis.processors.blip_processors import Blip2ImageTrainProcessor, BlipCaptionProcessor, BlipImageEvalProcessor, BlipImageTrainProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
to_tensor = transforms.transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
to_PIL = transforms.ToPILImage()
txt_processors = BlipCaptionProcessor(max_words=800)


def get_processors(model_type='clip'):
    if model_type == 'clip':
        _, vis_train_processors = clip.load("ViT-B/32", device=device, jit=False)
        vis_eval_processors = transforms.transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif model_type == 'blip':
        vis_train_processors = Blip2ImageTrainProcessor(image_size=224)
        vis_eval_processors = BlipImageEvalProcessor(image_size=224)
    else:
        vis_train_processors = BlipImageTrainProcessor()
        vis_eval_processors = BlipImageEvalProcessor(image_size=384)
    return vis_train_processors, vis_eval_processors


def is_valid_string(string):
    pattern = r'^[A-Za-z0-9 !"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]*$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False


class Pick_a_pic(Dataset):
    def __init__(self, data_path, vis_train_processors, vis_eval_processors, dataset_type='train'):
        table = pq.read_table(data_path)
        df = table.to_pandas()
        self.dataset_type = dataset_type
        self.vis_train_processors = vis_train_processors
        self.vis_eval_processors = vis_eval_processors
        if self.dataset_type == 'train':
            self.df = df[(df['has_label'] == True) & (df['caption'].apply(is_valid_string))]
        elif self.dataset_type == 'val':
            self.df = df[(df['has_label'] == True)]
        else:
            self.df = df
        print(data_path + " 加载完毕")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        raw_image1 = Image.open(io.BytesIO(self.df.iloc[item]['jpg_0'])).convert('RGB')
        raw_image2 = Image.open(io.BytesIO(self.df.iloc[item]['jpg_1'])).convert('RGB')

        if self.dataset_type == 'train':
            image1 = self.vis_train_processors(raw_image1)
            image2 = self.vis_train_processors(raw_image2)
        else:
            image1 = self.vis_eval_processors(raw_image1)
            image2 = self.vis_eval_processors(raw_image2)

        prompt = self.df.iloc[item]['caption']

        label1 = self.df.iloc[item]['label_0']
        label2 = self.df.iloc[item]['label_1']

        return image1, image2, txt_processors(prompt), torch.tensor([label1, label2]).float()


class train_data(Dataset):
    def __init__(self, data_root, vis_train_processors, vis_eval_processors, dataset_type='train', dataset_name='1'):
        path_list = os.listdir(data_root)
        self.prompts = []
        self.image1_path = []
        self.image2_path = []
        self.dataset_type = dataset_type
        self.vis_train_processors = vis_train_processors
        self.vis_eval_processors = vis_eval_processors
        for path in path_list:
            prompt_path = os.path.join(data_root, path, 'prompt.txt')
            prompt = open(prompt_path, encoding='utf-8').read(300)

            if self.dataset_type != 'test':
                image1_paths = glob.glob(data_root + path + '/good/*.png')
                image2_paths = glob.glob(data_root + path + '/bad/*.png')
                self.is_train = True
            else:
                image1_paths = glob.glob(data_root + path + '/image1/*.png')
                image2_paths = glob.glob(data_root + path + '/image2/*.png')
                self.is_train = False

            self.image1_path.extend(image1_paths)
            self.image2_path.extend(image2_paths)
            self.prompts.extend([prompt] * len(image1_paths))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        raw_image1 = Image.open(self.image1_path[item]).convert('RGB')
        raw_image2 = Image.open(self.image2_path[item]).convert('RGB')

        if self.dataset_type == 'train':
            image1_ = self.vis_train_processors(raw_image1)
            image2_ = self.vis_train_processors(raw_image2)
        else:
            image1_ = self.vis_eval_processors(raw_image1)
            image2_ = self.vis_eval_processors(raw_image2)

        if self.is_train:
            random_num = random.random()
            if random_num <= 0.5:
                image1 = image1_
                image2 = image2_
                label1 = 1.0
                label2 = 0.0
            else:
                image1 = image2_
                image2 = image1_
                label1 = 0.0
                label2 = 1.0

            return image1, image2, txt_processors(self.prompts[item]), torch.tensor([label1, label2]).float()
        else:
            image1 = image1_
            image2 = image2_
            label0 = 0.0
            label1 = 1.0

            return image1, image2, txt_processors(self.prompts[item]), torch.tensor([label0, label1]).float()


class HPDv1(Dataset):
    def __init__(self, data_path, vis_train_processors, vis_eval_processors, dataset_type='train'):
        self.prompts = []
        self.image0_path = []
        self.image1_path = []
        self.vis_train_processors = vis_train_processors
        self.vis_eval_processors = vis_eval_processors
        self.dataset_type = dataset_type
        if dataset_type != 'test':
            self.is_train = True
        else:
            self.is_train = False
        if dataset_type == 'train':
            data_file = 'preference_train.json'
        else:
            data_file = 'preference_test.json'
        with open(data_path + data_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        for c in content:
            good_path = c['file_path'][c['human_preference']]
            c['file_path'].remove(good_path)
            bad_path = c['file_path']
            for path in bad_path:
                self.image0_path.append(data_path + good_path)
                self.image1_path.append(data_path + path)
                self.prompts.append(c['prompt'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        image0_path = self.image0_path[item]
        raw_image0 = Image.open(image0_path).convert('RGB')

        image1_path = self.image1_path[item]
        raw_image1 = Image.open(image1_path).convert('RGB')

        if self.dataset_type == 'train':
            image0 = self.vis_train_processors(raw_image0)
            image1 = self.vis_train_processors(raw_image1)
        else:
            image0 = self.vis_eval_processors(raw_image0)
            image1 = self.vis_eval_processors(raw_image1)

        if self.is_train:
            good_image = image0
            bad_image = image1
            raw_good_image = to_tensor(raw_image0)
            raw_bad_image = to_tensor(raw_image1)

            random_num = random.random()
            if random_num <= 0.5:
                image0_p = raw_good_image
                image1_p = raw_bad_image
                label0 = 1.0
                label1 = 0.0
            else:
                image0_p = raw_bad_image
                image1_p = raw_good_image
                label0 = 0.0
                label1 = 1.0
            return good_image, bad_image, txt_processors(self.prompts[item]), \
                   image0_p, image1_p, torch.tensor([label0, label1]).float()
        else:
            label0 = 1
            label1 = 0
            image0_p = to_tensor(raw_image0)
            image1_p = to_tensor(raw_image1)

            return image0, image1, txt_processors(self.prompts[item]), \
                   image0_p, image1_p, torch.tensor([label0, label1]).float()


def get_dataloader(data_root, batch_size, data_set, num_workers=0, dataset_type='train', model_type='clip'):
    vis_train_processors, vis_eval_processors = get_processors(model_type)
    if dataset_type == 'train':
        is_shuffle = True
    else:
        is_shuffle = False
    if data_set == 'Pick-a-pic':
        path_list = os.listdir(data_root)
        path_list = sorted(path_list)
        datasets = Pick_a_pic(data_root + '/' + path_list[0], vis_train_processors, vis_eval_processors, dataset_type)
        for i in range(1, len(path_list)):
            datasets += Pick_a_pic(data_root + '/' + path_list[i], vis_train_processors, vis_eval_processors, dataset_type)
    elif data_set == 'HPDv1':
        datasets = HPDv1(data_root, vis_train_processors, vis_eval_processors, dataset_type)
    elif data_set == 'ImageR':
        datasets = train_data(data_root, vis_train_processors, vis_eval_processors, dataset_type, data_set)
    else:
        datasets = train_data(data_root, vis_train_processors, vis_eval_processors, dataset_type, data_set)
        # if dataset_type == 'train':
        #     datasets, _ = random_split(dataset=datasets, lengths=[4832, 1208], generator=torch.manual_seed(42))
        # elif dataset_type == 'val':
        #     _, datasets = random_split(dataset=datasets, lengths=[4832, 1208], generator=torch.manual_seed(42))

    dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    return dataloader
