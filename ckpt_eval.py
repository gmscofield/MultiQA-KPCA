import os
import argparse
import pickle
import torch

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

from evaluation import test_evaluation
from model import MyModel
from dataloader import load_t1_data

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size",type=int,default=300)
    parser.add_argument("--overlap",type=int,default=45)
    parser.add_argument("--checkpoint_path",default="/home/jiachangguo/Multi-Turn-QA-master/checkpoints/ace2005/2022_06_05_00_18_50/checkpoint_0.cpt")
    parser.add_argument("--test_path",default="/home/jiachangguo/Multi-Turn-QA-master/data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_300_threshold_5_max_distance_45/test.json")
    parser.add_argument("--test_batch",default=10,type=int)
    parser.add_argument('--max_len',default=512,type=int)
    parser.add_argument("--threshold",type=int,default=-1)
    parser.add_argument("--amp",action='store_true')
    args = parser.parse_args()
    model_dir,file = os.path.split(args.checkpoint_path)
    config = pickle.load(open(os.path.join(model_dir,'args'),'rb'))
    checkpoint = torch.load(os.path.join(model_dir,file),map_location=torch.device("cpu"))
    model_state_dict = checkpoint['model_state_dict']
    config.pretrained_model_path = args.pretrained_model_path if args.pretrained_model_path else config.pretrained_model_path
    config.threshold = args.threshold if args.threshold==-1 else config.threshold
    mymodel = MyModel(config)
    mymodel.load_state_dict(model_state_dict,strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mymodel.to(device)
    
    # test_dataloader = load_t1_data(config.dataset_tag,args.test_path,config.pretrained_model_path,args.window_size,args.overlap,args.test_batch,args.max_len)
    test_dataloader = load_t1_data(config.dataset_tag,args.test_path,args.window_size,args.overlap,args.test_batch,args.max_len)
    (p1,r1,f1),(p2,r2,f2) = test_evaluation(mymodel,test_dataloader,config.threshold,args.amp)
    print("Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1,r1,f1))
    print("Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2,r2,f2))