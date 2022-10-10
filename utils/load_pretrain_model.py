# -*- coding: utf-8 -*-
"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:02/12/2021
"""

import torch
import torchvision
import os
import stat
import numpy as np


def load_model(dataset_name):
    print("--------------------load_model----------------------------")
    model = torchvision.models.resnet50(pretrained=False)
    if dataset_name == 'CUB':
        # chapter_4
        # 路径有点变动 cxh
        #path_model = 'G:\\github_files\\FGVC_platform\models\\cub_pgd_1200ep_448_best.pth.tar'  # the first pretrained model
        #path_model = 'E:\\cub数据集\\CUB\\test\\001.Black_footed_Albatross\\Black_Footed_Albatross_0001_796111.jpg'
        path_model = 'D:\\model\\cub_pgd_1200ep_448_best.pth.tar'
        # change output_features
        num_fc_ftr = model.fc.in_features
        print("num_fc_ftr",num_fc_ftr)
        model.fc = torch.nn.Linear(num_fc_ftr, 200)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    elif dataset_name == 'CAR':  # 这里代码都有问题，因为和 car 和 aircraft 模型不匹配
        # chapter_4
        #这里路径改动 cxh
        #path_model = 'G:\\github_files\\FGVC_platform\\models\\pgd_car_1200ep_448_best.pth.tar'
        #path_model = 'E:\\Granularity_portfolio\\cars_train'
        path_model = 'E:\\cub\\CUB\\test'
        # change output_features
        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, 196)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    elif dataset_name == 'Aircraft':
        # chapter_4
        path_model = 'G:\\github_files\\FGVC_platform\\models\\aircraft_pgd_1200_448_best.tar'  # the first pretrained model

        # change output_features
        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, 100)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    else:
        #原本的代码更改，看效果
         raise ('not support dataset!!!')
        #以下是增加的代码，以上是原本的代码
        # print("----------在这里-----------------")
        # # path_model = "E:\\Granularity_portfolio\\cars_test"
        # #path_model = "E://Granularity_portfolio//cars_test"
        # path_model = "E:\\cub数据集\\CUB\\test"
        # # filesname = 'Granularity_portfolio'
        # # GPfilename = 'E:\\Granularity_portfolio'
        # # os.chmod(filesname, stat.S_IWRITE)#取消文件只读属性
        # # print('------文件只读属性取消--------')
        # # change output_features
        # num_fc_ftr = model.fc.in_features
        # model.fc = torch.nn.Linear(num_fc_ftr, 196)
        # checkpoint = torch.load(path_model, map_location="cpu")
        # state_dict = checkpoint['state_dict']
#=-----------------------------------------------------------------------------------------------------------------------
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    msg = model.load_state_dict(state_dict, strict=False)
    print(set(msg.missing_keys))
    return model
