import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Module import CAM, FeatMap
from Dataset import DataLoader

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='/home/hexianhao/Workspace/Python/MultiAttNet_center_loss/images', 
                        help='the file path of images')
    parser.add_argument('--model_path', type=str, default='/home/hexianhao/Workspace/Python/Few_Shot_CAM/few_shot_model.pkl')
    parser.add_argument('--cuda', type=bool, default=True, help='enable cuda')
    parser.add_argument('--disp_step', type=int, default=10,
                        help='display step during training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--max_step', type=int, default=10000, help='max step during training')
    parser.add_argument('--test_step', type=int, default=50, help='test times')
    parser.add_argument('--img_size', type=int, default=224, help='the size of image fed into CNN')
    parser.add_argument('--n_way', type=int, default=5, help='n way during test')
    parser.add_argument('--k_shot', type=int, default=1, help='k shot during test')
    parser.add_argument('--n_category', type=int, default=40, help='num class in training batch')
    parser.add_argument('--n_samples', type=int, default=4, help='images for each class in training batch')
    parser.add_argument('--test_size', type=int, default=16, help='batch size for test')
    
    arg_opt = parser.parse_args()
    dataloader = DataLoader(arg_opt.image_file)

    resnet = FeatMap(100)
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.SGD(trans.parameters(), lr=arg_opt.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-8)
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    pretrain_model = torch.load("/home/hexianhao/Workspace/Python/Few_Shot_CAM/model.pkl")
    pretrain_dict = pretrain_model.state_dict()
    model_dict = resnet.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    resnet.load_state_dict(model_dict)

    softmax_weights = resnet.fc.weight.data.numpy()