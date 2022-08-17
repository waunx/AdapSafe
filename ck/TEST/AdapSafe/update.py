# coding=utf-8

import json
# 导入json头文件

import os, sys

json_path = './freq_control_mql_22-43-15.json'
# json原文件

json_path1 = './freq_control_mql_22-43-15.json'
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
            "lr": 0.0005,
            "replay_size": 1000000.0,
            "ptau": 0.005,
            "gamma": 0.99,
            "burn_in": 10000.0,
            "total_timesteps": 10000000.0,
            "expl_noise": 0.0,
            "batch_size": 256,
            "policy_noise": 0.0,
            "noise_clip": 0.0,
            "policy_freq": 2,
            "hidden_sizes": [
                400,
                100
            ],
            "env_name": "freq_control",
            "seed": 29,
            "alg_name": "adapsafe",
            "disable_cuda": False,
            "cuda_deterministic": False,
            "gpu_id": 5,
            "check_point_dir": "./ck",
            "log_dir": "./log_dir/2022-07-31",
            "log_interval": 10,
            "save_freq": 250,
            "eval_freq": 30000.0,
            "env_configs": "./configs/pearl_envs.json",
            "max_path_length": 200,
            "enable_train_eval": False,
            "enable_promp_envs": False,
            "num_initial_steps": 1000,
            "unbounded_eval_hist": False,
            "hiddens_conext": [
                30
            ],
            "enable_context": True,
            "only_concat_context": 3,
            "num_tasks_sample": 5,
            "num_train_steps": 1000,
            "min_buffer_size": 100000,
            "history_length": 10,
            "beta_clip": 2.0,
            "snapshot_size": 2000,
            "prox_coef": 0.1,
            "meta_batch_size": 10,
            "enable_adaptation": True,
            "main_snap_iter_nums": 200,
            "snap_iter_nums": 100,
            "type_of_training": "td3",
            "lam_csc": 0.1,
            "use_ess_clipping": True,
            "enable_beta_obs_cxt": True,
            "sampling_style": "replay",
            "sample_mult": 5,
            "use_epi_len_steps": True,
            "use_normalized_beta": False,
            "reset_optims": False,
            "lr_milestone": 5000000,
            "lr_gamma": 0.8,
            "test": False,
            "max_test_episode": 4,
            "pre_train": False,
            "enable_CBF": True,
            "enable_GP": False,
            "CBF_name": "ZCBF",
            "CBF_penalty": 20,
            "n_train_tasks": 50,
            "n_eval_tasks": 10,
            "n_tasks": 60,
            "randomize_tasks": True,
            "low_gear": False,
            "forward_backward": True,
            "num_evals": 5,
            "num_steps_per_task": 200,
            "num_steps_per_eval": 200,
            "num_train_steps_per_itr": 4000,
            "output_dim_conext": 30,
            "adapt_csv_hearder": {
                "eps_num": None,
                "iter": None,
                "critic_loss": None,
                "actor_loss": None,
                "csc_samples_neg": None,
                "csc_samples_pos": None,
                "train_acc": None,
                "snap_iter": None,
                "beta_score": None,
                "main_critic_loss": None,
                "main_actor_loss": None,
                "main_beta_score": None,
                "main_prox_critic": None,
                "main_prox_actor": None,
                "main_avg_prox_coef": None,
                "tidx": None,
                "avg_rewards": None,
                "one_raw_reward": None,
                "ep_nadir": None,
                "ep_gamma": None,
                "mean_ep_action": None
            }
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
