"""Script for running caption on custom data."""
import argparse
import os
#import ruamel_yaml as yaml
import yaml
import numpy as np
import random
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
from models.model_caption_mplug import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils, visualizer


class SimpleDataset(torch.utils.data.Dataset):
    """
    Image-only datasets.
    """
    def __init__(self, data_path, size=None, transform=None):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        self.files = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(data_path) if files], [])
        self.files.sort()

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(os.path.join(self.data_path, fpath), "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.size:
                img = img.resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)



def load_model(model, args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint['module']

    # reshape positional embedding to accomodate for image resolution change
    if config["clip_name"] == "ViT-B-16":
        num_patches = int(config["image_res"] * config["image_res"]/(16*16))
    elif config["clip_name"] == "ViT-L-14":
        num_patches = int(config["image_res"] * config["image_res"]/(14*14))
    pos_embed = torch.nn.Parameter(torch.zeros(num_patches + 1, 768).float())

    pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                pos_embed.unsqueeze(0))
    state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

    #if False: #not args.evaluate:
    #for key in list(state_dict.keys()):
    #    if ('fusion' in key or 'bert' in key) and 'decode' not in key:
    #        encoder_key = key.replace('fusion.', '').replace('bert.', '')
    #        state_dict[encoder_key] = state_dict[key]
    #        del state_dict[key]


    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % args.checkpoint)
    print(msg)



def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.Resampling.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ]) 
    data_dir = "/home/jianjin/data/SDE/data/celebahq/image"
    dataset = SimpleDataset(data_dir, transform=test_transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)
    model = model.to(device)
    load_model(model, args)

    n_col = 8
    n_row = 5
    page = visualizer.HtmlPageVisualizer(n_row, n_col)

    result = []
    for i, image in enumerate(data_loader):        
        image = image.to(device, non_blocking=True)   
        question_input = [config['bos']] * len(caption)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        topk_ids, topk_probs = model(image, question, train=False, scst=args.scst)
        for topk_id, topk_prob in zip(topk_ids, topk_probs):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append(ans)
        print(ans)
        disp = (image - image.min()) / (image.max() - image.min())
        disp = disp.cpu().numpy()[0].transpose(1, 2, 0)
        disp = (disp * 255).astype("uint8")
        row_idx, col_idx = i // n_col, i % n_col
        page.set_cell(row_idx, col_idx, text=ans, image=disp)
        if i >= n_row * n_col - 1:
            break
    page.save("res.html")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_mplug_large.yaml')
    parser.add_argument('--checkpoint', default='/home/jianjin/data/SDE/pretrained/mplug_large_v2.pth')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    #config['optimizer']['lr'] = args.lr
    #config['schedular']['lr'] = args.lr
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder


    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)