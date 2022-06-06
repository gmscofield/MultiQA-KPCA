import os
import time
import random
import argparse
import pickle
from transformers.optimization import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# import sys
# import os
#
# model_dir = os.path.dirname(os.path.dirname("model.py"))
# if not model_dir in sys.path:
#     sys.path.append(model_dir)
#
# eval_dir = os.path.dirname(os.path.dirname("evaluation.py"))
# if not eval_dir in sys.path:
#     sys.path.append(eval_dir)
#
# data_dir = os.path.dirname(os.path.dirname("dataloader.py"))
# if not data_dir in sys.path:
#     sys.path.append(data_dir)

from model import MyModel
from evaluation import test_evaluation
from dataloader import load_data, load_t1_data, reload_data


#train.py
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",default="./data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_300_threshold_5_max_distance_45/train.json")
    parser.add_argument("--train_batch", type=int, default=10)
    parser.add_argument("--test_path",default="./data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_300_threshold_5_max_distance_45/test.json")
    parser.add_argument("--test_batch", type=int, default=10)
    parser.add_argument("--max_len", default=512, type=int,
                        help="maximum length of input")
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--warmup_ratio", type=float, default=-1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--theta", type=float,
                        help="weight of two tasks", default=0.25)
    parser.add_argument("--window_size", type=int,
                        default=100, help="size of the sliding window")
    parser.add_argument("--overlap", type=int, default=50,
                        help="overlap size of the two sliding windows")
    parser.add_argument("--threshold", type=int, default=5,
                        help="the limitation")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true",
                        help="whether to enable mixed precision")
    args = parser.parse_args()
    return args


def train(args, train_dataloader):
    model = MyModel(args)
    model.train()
    scaler = GradScaler()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, output_device=device, find_unused_parameters=True)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay":args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay":0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    if args.warmup_ratio > 0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        warmup_steps = args.warmup_ratio*num_training_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps)
    for epoch in range(args.max_epochs):
        tqdm_train_dataloader = tqdm(
            train_dataloader, desc="epoch:%d" % epoch, ncols=150)
        for i, batch in enumerate(tqdm_train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, tags = batch['txt_ids'], batch['attention_mask'], batch['token_type_ids'],\
                batch['context_mask'], batch['turn_mask'], batch['tags']
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, tags = txt_ids.to(device), attention_mask.to(device), token_type_ids.to(device),\
                context_mask.to(device), turn_mask.to(device), tags.to(device)
            if args.amp:
                with autocast():
                    loss, (loss_t1, loss_t2) = model(
                        txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, tags)
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, (loss_t1, loss_t2) = model(txt_ids, attention_mask,
                                                 token_type_ids, context_mask, turn_mask, tags)
                loss.backward()
                if args.max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            named_parameters = [
                (n, p) for n, p in model.named_parameters() if not p.grad is None]
            grad_norm = torch.norm(torch.stack(
                [torch.norm(p.grad) for n, p in named_parameters])).item()
            if args.warmup_ratio > 0:
                scheduler.step()
            postfix_str = "norm:{:.2f},lr:{:.1e},loss:{:.2e},t1:{:.2e},t2:{:.2e}".format(
                grad_norm, lr, loss.item(), loss_t1, loss_t2)
            tqdm_train_dataloader.set_postfix_str(postfix_str)
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        checkpoint = {"model_state_dict": model_state_dict}
        save_dir = './checkpoints/%s/%s/' % ("ace2005")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            pickle.dump(args, open(save_dir+'args', 'wb'))
        save_path = save_dir+"checkpoint_%d.cpt" % epoch
        torch.save(checkpoint, save_path)
        print("model saved at:", save_path)

        test_dataloader = load_t1_data(args.test_path,
                                           args.window_size, args.overlap, args.test_batch, args.max_len)  # test_dataloader是第一轮问答的dataloder
        (p1, r1, f1), (p2, r2, f2) = test_evaluation(
            model, test_dataloader, args.threshold, args.amp)
        print("Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1, r1, f1))
        print("Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2, r2, f2))
        model.train()




if __name__ == "__main__":
    args = args_parser()
    set_seed(args.seed)
    print(args)
    p = '{}_{}_{}'.format("ace2005", os.path.split(
        args.train_path)[-1].split('.')[0], "bert-base-uncased")
    p1 = os.path.join(os.path.split(args.train_path)[0], p)
    if not os.path.exists(p1):
        train_dataloader = load_data("ace2005", args.train_path, args.train_batch, args.max_len,
                                     shuffle=True, threshold=args.threshold)
        pickle.dump(train_dataloader, open(p1, 'wb'))
        print("training data saved at ", p1)
    else:
        print("reload training data from ", p1)
        train_dataloader = pickle.load(open(p1, 'rb'))
        train_dataloader = reload_data(train_dataloader, args.train_batch, args.max_len,args.threshold)
        pickle.dump(train_dataloader, open(p1, 'wb'))
    train(args, train_dataloader)
