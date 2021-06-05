"""
 基于混合图谱的对话推荐系统
 社交网络挖掘
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse


class TrainModel():
    def __init__(self, opt):
        # TODO: Model part

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-makedata","--makedata",type=bool,default=False)
    argParser.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
    argParser.add_argument("-epoch","--epoch",type=int,default=30)
    argParser.add_argument("-batch_size","--batch_size",type=int,default=16)
    args = argParser.parse_args()
    # Print the arguments
    print(vars(args))
    if args.makedata==True:
        # Data prepossessing part
    if args.is_finetune==False:
        # Pre-training
        kgsf = TrainModel(vars(args))
        kgsf.train()
    else:
        # Fine-tuning



