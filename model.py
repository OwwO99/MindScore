import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import clip


block_list = ['blocks.38', 'blocks.37', 'blocks.36', 'blocks.35', 'blocks.34']
block_list1 = ['blocks.11', 'blocks.10', 'blocks.9', 'blocks.8', 'blocks.7']


class ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        return self.model.encode_image(img)


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class Net(nn.Module):
    def __init__(self, img_encoder, text_encoder, projection) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.projection = projection

    def forward(self, img_1, img_2, prompt):
        img_1_embedding = self.norm(self.img_encoder(img_1))
        img_2_embedding = self.norm(self.img_encoder(img_2))
        prompt_embedding = self.norm(self.text_encoder(prompt))
        return self.projection(img_1_embedding), self.projection(img_2_embedding), self.projection(prompt_embedding)

    def norm(self, vec: torch.Tensor):
        return vec / vec.norm(dim=1, keepdim=True)


class ImageEncoder_blip1(nn.Module):
    def __init__(self, visual_encoder, vision_proj, layers=True):
        super(ImageEncoder_blip1, self).__init__()
        self.visual_encoder = visual_encoder
        for p in self.parameters():
            p.requires_grad = False
        if layers:
            for name, param in self.visual_encoder.named_parameters():
                if name[: 9] in block_list1 or name[: 8] in block_list1:
                    param.requires_grad = True
        self.vision_proj = vision_proj

    def forward(self, image):
        image_embeds = self.visual_encoder.forward_features(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        return image_feat


class TextEncoder_blip1(nn.Module):
    def __init__(self, text_encoder, text_proj):
        super(TextEncoder_blip1, self).__init__()
        self.text_encoder = text_encoder
        self.text_proj = text_proj

    def forward(self, input_ids, attention_mask):
        text_output = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            mode="text",
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return text_feat


class ImageEncoder_blip(nn.Module):
    def __init__(self, visual_encoder, ln_vision, query_tokens, bert, vision_proj, layers=True):
        super().__init__()
        self.visual_encoder = visual_encoder
        for p in self.parameters():
            p.requires_grad = False
        if layers:
            for name, param in self.visual_encoder.named_parameters():
                if name[: 9] in block_list:
                    param.requires_grad = True
        self.bert = bert
        self.ln_vision = ln_vision
        self.query_tokens = query_tokens
        self.vision_proj = vision_proj

    def forward(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image)).float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # [b, 32, 768] [b, 768]
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state[:, 0, :]), dim=-1
        )
        # [b, 256]
        return image_feats


class TextEncoder_blip(nn.Module):
    def __init__(self, bert, text_proj):
        super().__init__()
        self.bert = bert
        self.text_proj = text_proj

    def forward(self, input_ids, attention_mask):
        text_output = self.bert(
            input_ids,
            attention_mask,
            return_dict=True,
        )
        # [b, 256, 768] [b, 768]
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        # [b, 256]
        return text_feat


class Net_blip(nn.Module):
    def __init__(self, img_encoder, text_encoder) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder

    def forward(self, img_1, img_2, input_ids, attention_mask):
        # [b, 3, 364, 364]
        img_1_embedding = self.img_encoder(img_1)
        img_2_embedding = self.img_encoder(img_2)
        prompt_embedding = self.text_encoder(input_ids, attention_mask)

        return img_1_embedding, img_2_embedding, prompt_embedding


class Projection(nn.Module):
    def __init__(self, num_hidden=512) -> None:
        super().__init__()
        self.linear1 = nn.Linear(num_hidden, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.activation = F.relu

    def forward(self, embedding):
        return self.linear2(self.activation(self.linear1(embedding.to(self.linear1.weight.dtype))))


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=pretrained)
        # print(list(resnet.children()))
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.res1 = resnet.layer1
        self.res2 = resnet.layer2
        self.res3 = resnet.layer3
        self.res4 = resnet.layer4
        self.last_dim = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return x


class FeatureExtractor_VIT(nn.Module):
    def __init__(self, mode, pretrained=True):
        super(FeatureExtractor_VIT, self).__init__()
        if mode == 'blip1':
            vit = timm.create_model('vit_base_patch16_384', pretrained=pretrained)
            self.h = 24
        else:
            vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            self.h = 14
        self.patch_embedding = vit.patch_embed
        self.pos_embedding = vit.pos_embed
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.last_dim = vit.head.in_features

    def forward(self, x):
        x = self.patch_embedding(x)
        # [b, 196, 768]
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        # [b, 196, 768]
        x = self.blocks(x)
        # [b, 196, 768]
        x = self.norm(x)
        # [b, 196, 768]
        b, fs, embed_dim = x.shape
        x = x.view(b, embed_dim, self.h, self.h)
        # [b, 768, 14, 14]
        return x


class GlobalAndLocalPooling(nn.Module):
    def __init__(self, output_size=(2, 2)):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor):
        local_pooled = self.local_pool(x)
        global_pooled = self.global_pool(x)
        N, C, oh, ow = local_pooled.size()
        local_pooled = local_pooled.permute(0, 2, 3, 1)
        local_pooled = local_pooled.reshape((N, oh * ow, C))
        global_pooled = global_pooled.permute(0, 2, 3, 1).reshape((N, 1, C))
        aggregate_pooled = torch.cat([local_pooled, global_pooled], dim=1)
        # [b, 5, 768] [b, 3840]
        return aggregate_pooled.reshape((N, -1))


class PerceptualNet(nn.Module):
    def __init__(self, mode, pretrained=True, pool_window=(2, 2), fc_dims=(1024, 256), num_classes=2):
        super(PerceptualNet, self).__init__()
        self.base = FeatureExtractor_VIT(mode=mode, pretrained=pretrained)
        self.head = self.get_head(pool_window, fc_dims, num_classes)

    def get_head(self, pool_window, fc_dims, num_classes):
        in_dim = self.base.last_dim * 3 * (pool_window[0] * pool_window[1] + 1)  # 3840
        global_and_local_pool = GlobalAndLocalPooling(pool_window)
        fc1 = nn.Linear(in_features=in_dim, out_features=fc_dims[0])
        fc2 = nn.Linear(in_features=fc_dims[0], out_features=fc_dims[1])
        fc3 = nn.Linear(in_features=fc_dims[1], out_features=num_classes)
        return nn.Sequential(global_and_local_pool, fc1, fc2, fc3)

    def forward(self, x1, x2):
        ft1 = self.base(x1)
        ft2 = self.base(x2)
        # [N, 768, 14, 14]
        ft = torch.concat([ft1, ft2, ft1 - ft2], dim=1)
        output = self.head(ft)
        return output  # F.softmax(self.head(ft), dim=-1)


class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()
        self.model, self.preprocess = clip.load(name, device="cpu")  # self.preprecess will not be used during training, which is handled in Dataset class
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join('./checkpoints', 'experiment_name')
        self.device = torch.device('cuda:{}'.format('0')) if '0' else torch.device('cpu')

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }

        torch.save(state_dict, save_path)


    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


# Realness
class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = CLIPModel("ViT-L/14")
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)
        params = self.model.parameters()

        self.optimizer = torch.optim.AdamW(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to('cuda:0')

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()