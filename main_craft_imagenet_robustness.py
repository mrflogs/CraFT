import os
import random
import argparse
import yaml

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import transformers

import numpy as np

from datasets.imagenet_r import ImageNet_R
from datasets.imagenet_sketch import ImageNet_Sketch
from datasets.imagenet_a import ImageNet_A
from datasets.imagenet_v2 import ImageNet_V2
from datasets.imagenet import ImageNet

from utils import *
import cma
import time

_tokenizer = _Tokenizer()
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.type(self.dtype)
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, train_loader_F):
        super().__init__()
        self.train_features, self.train_labels = pre_load_features(cfg, "train", clip_model, train_loader_F)
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = 'a photo of a' 
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].cuda()
        prompt_prefix = ctx_init       
        self.n_ctx = n_ctx

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = ctx_vectors
        self.bias = torch.zeros_like(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, clip_model)
        self.linear = nn.Linear(512, n_ctx * ctx_vectors.shape[1], bias=False)
        
    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.text_encoder = TextEncoder(clip_model).float().cuda()
        self.name_lens = name_lens

    def forward(self, prompt_embedding=None, net=None, weight=0.1, steps=0):
        '''
        State Control:
        - prompt_embedding == None: Tunable head optimization.
        - prompt_embedding != None: CMA-ES optimization.
        '''         
        # tunable head optimization.
        if prompt_embedding is None:
            ctx = self.ctx # p_0
            ctx = ctx + self.bias
            
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        
            text_features = self.text_encoder(prompts.float(), self.tokenized_prompts.float())
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)        
            return text_features 
            
        # cma-es optimization.
        else:
            ctx = self.ctx # p_0
            ctx = ctx + self.linear(prompt_embedding).reshape(self.n_ctx, -1) # p_0 + Az
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            with torch.no_grad():
                text_features = self.text_encoder(prompts.float(), self.tokenized_prompts.float())
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
                logits = 100. * self.train_features.float() @ text_features.T.float()
            update_logits = logits + net(logits)
            loss = F.cross_entropy(logits, self.train_labels.long()) 
            loss += weight * ce_loss(logits, update_logits) 
            return loss.unsqueeze(0)
        
    def update_context_prompt(self, es):
        self.bias = self.linear(torch.tensor(es.result.xbest).cuda().float()).reshape(self.n_ctx, -1)
        
def run(cfg, dataset, clip_model, test_loader, train_loader_F):
    # Pre-load val features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)  
    train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)   
    train_features, train_labels = pre_load_features(cfg, "train", clip_model, train_loader) 
    
    cma_opts = {
        'seed': cfg["seed"],
        'popsize': cfg["popsize"],
        'maxiter': cfg["budget"] if cfg["parallel"] else cfg["budget"] // cfg["popsize"],
        'verbose': -1,
    }
    
    if cfg["bound"] > 0:
        cma_opts['bounds'] = [-1 * cfg["bound"], 1 * cfg["bound"]]
    es = cma.CMAEvolutionStrategy(cfg["intrinsic_dim"] * [0], cfg["sigma"], inopts=cma_opts)
    print('Population Size: {}'.format(es.popsize))
    print('{} Evaluation.'.format('Parallel' if cfg["parallel"] else 'Serial'))
    
    prompt_learner = PromptLearner(cfg, dataset.classnames, clip_model, train_loader_F).cuda()
    net = nn.Sequential(
        nn.Linear(len(dataset.classnames), 512),
        nn.ReLU(),
        nn.Linear(512, len(dataset.classnames))
    ).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    
    steps = 0
    w = cfg["weight"]
    
    while not es.stop():
        net.train()
        solutions = es.ask() # list of numpy array. [numpy.ndarray]. len(solutions) = cfg["popsize"]
        inputs = torch.tensor(np.array(solutions)).cuda().float()
        # update prompt
        with torch.no_grad():
            losses = [prompt_learner(x, net, weight=cfg["weight"], steps=steps) for x in inputs]
            fitnesses = [loss.item() for loss in losses]
            es.tell(solutions, fitnesses)
            es.disp()
        prompt_learner.update_context_prompt(es)
        steps += cfg["popsize"]
        
        # update head. (可以替换为 cma-es update.)
        with torch.no_grad():
            text_features = prompt_learner()
        for image, target in train_loader_F:
            
            with torch.no_grad():
                image, target = image.cuda(), target.cuda()
                image_features = clip_model.encode_image(image)
                logits = 100. * image_features.float() @ text_features.T.float()
            update_logits = logits + net(logits)
            cls_loss = F.cross_entropy(update_logits, target.long())
            ce = ce_loss(update_logits, logits)
            loss = cls_loss + (w if (steps <= cfg["budget"]//cfg["n"]) else 0) * ce # 分布约束
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # evaluate
    with torch.no_grad():
        net.eval()
        text_features = prompt_learner()
        logits = 100. * test_features.float() @ text_features.T.float()
        logits = logits + net(logits)
        test_logits = logits

        acc = cls_acc(test_logits, test_labels)
        print("Acc:%s" % (acc))
    return acc, prompt_learner, net
    
def ce_loss(input_logits, target_logits):
    target_dist = torch.softmax(target_logits, dim=1)
    input_dist = F.log_softmax(input_logits, dim=1)
    loss = F.kl_div(input_dist, target_dist, reduction='batchmean')
    return loss

def main():
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
     
    cfg['subsample_classes'] = 'all'
    
    cfg["budget"] = 2000
    
    # ---------------------------------------- run ------------------------------------
    
    for backbone in ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16']:
        cfg["backbone"] = backbone
        # load CLIP
        global preprocess
        clip_model, preprocess = clip.load(cfg['backbone'])
        clip_model.eval() 
        results = {"1": [], "2": [], "3": []}
        for seed in [1, 2, 3]:
            cfg['seed'] = seed
            print("\nRunning configs.")
            print(cfg, "\n")
            
            random.seed(cfg['seed'])
            torch.manual_seed(cfg['seed'])
            print("Preparing dataset.")

            dataset_shift = [ImageNet_R(cfg['root_path'], preprocess), ImageNet_A(cfg['root_path'], preprocess), 
                             ImageNet_Sketch(cfg['root_path'], preprocess), ImageNet_V2(cfg['root_path'], preprocess)]
            dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
            classnames = dataset.classnames
            
            train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=True)      
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False)   
            
            acc, prompt_learner, net = run(cfg, dataset, clip_model, test_loader, train_loader_F)
            results[str(seed)].append(acc)
            for dataset in dataset_shift:
                label_mapping = dataset.label_mapping
                test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False)   
                test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)  
                # evaluate
                with torch.no_grad():
                    net.eval()
                    text_features = prompt_learner()
                    logits = 100. * test_features.float() @ text_features.T.float()
                    logits = logits + net(logits)
                    test_logits = logits
                    test_logits = test_logits @ label_mapping
                    acc = cls_acc(test_logits, test_labels)
                    results[str(seed)].append(acc)
            
        print("Dataset: %s" % (cfg["dataset"]))
        print("Resutls on backbone:", backbone)
        print("Results on shots: [16]")
        for seed in ["1", "2", "3"]:
            print("Results on seed %s: %s" % (seed, results[seed]))
        print("Average results:", torch.tensor([results["1"], results["2"], results["3"]]).mean(dim=0))

if __name__ == '__main__':
    main()
