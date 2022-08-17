# coding=utf-8

import json
# 导入json头文件

import os, sys

json_path = './pearl_envs.json'
# json原文件

json_path1 = './pearl_envs.json'
# 修改json文件后保存的路径

dict = {}


# 用来存储数据

def get_json_data(json_path):
    # 获取json里面数据

    with open(json_path, 'rb') as f:
        # 定义为只读模型，并定义名称为f

        params = json.load(f)
        # 加载json文件中的内容给params

        params['freq_control'] = {
            "n_train_tasks": 50,
            "n_eval_tasks": 10,
            "n_tasks": 60,
            "randomize_tasks": True,
            "low_gear": False,
            "forward_backward": True,
            "num_evals": 5,
            "num_steps_per_task": 200,
            "num_steps_per_eval": 200,
            "num_train_steps_per_itr": 4000
        }
        # 修改内容

        print("params", params)
        # 打印
        dict = params
        # 将修改后的内容保存在dict中

    f.close()
    # 关闭json读模式

    return dict
    # 返回dict字典内容


def write_json_data(dict):
    # 写入json文件

    with open(json_path1, 'w') as r:
        # 定义为写模式，名称定义为r
        # indent=1表示自动换行！！！
        json.dump(dict, r, indent=1)
        # 将dict写入名称为r的文件中

    r.close()
    # 关闭json写模式


the_revised_dict = get_json_data(json_path)
write_json_data(the_revised_dict)
# 调用两个函数，更新内容
