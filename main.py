import argparse
import torch
import os
import time
import sys
import numpy as np
from collections import deque
import random
from misc.utils import create_dir, dump_to_json, CSVWriter
from misc.torch_utility import get_state
from misc.utils import safemean, read_json
from misc import logger
from algs.AdapSafe.buffer import Buffer
from safe.cbf import CBFController
from safe.dynamics_gp import DynamicGP

import warnings

warnings.filterwarnings("ignore", category=Warning)

parser = argparse.ArgumentParser()

# Optim params
parser.add_argument('--lr', type=float, default=0.0008, help='Learning rate')
parser.add_argument('--replay_size', type=int, default=1e6, help='Replay buffer size int(1e6)')
parser.add_argument('--ptau', type=float, default=0.005, help='Interpolation factor in polyak averaging')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor [0,1]')
parser.add_argument("--burn_in", default=1e4, type=int, help='How many time steps purely random policy is run for')
parser.add_argument("--total_timesteps", default=1e5, type=float, help='Total number of timesteps to train on')
parser.add_argument("--expl_noise", default=0.3, type=float, help='Std of Gaussian exploration noise')
parser.add_argument("--batch_size", default=256, type=int, help='Batch size for both actor and critic')
parser.add_argument("--policy_noise", default=0.3, type=float, help=' Noise added to target policy during critic update')
parser.add_argument("--noise_clip", default=0.0, type=float, help='Range to clip target policy noise')
parser.add_argument("--policy_freq", default=2, type=int, help='Frequency of delayed policy updates')
parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[300, 300], help='indicates hidden size actor/critic')

# General params
parser.add_argument('--env_name', type=str, default='freq_control')
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--alg_name', type=str, default='adapsafe')

parser.add_argument('--disable_cuda', default=False, action='store_true')
parser.add_argument('--cuda_deterministic', default=False, action='store_true')
parser.add_argument("--gpu_id", default=4, type=int)

parser.add_argument('--check_point_dir', default='./ck')
parser.add_argument('--log_dir', default=f"./log_dir/{time.strftime('%Y-%m-%d')}")
parser.add_argument('--log_interval', type=int, default=10, help='log interval, one log per n updates')
parser.add_argument('--save_freq', type=int, default=250)
parser.add_argument("--eval_freq", default=3e4, type=float, help='How often (time steps) we evaluate')
parser.add_argument('--pre_train', default=False, action='store_true')

# Env
parser.add_argument('--env_configs', default='./configs/pearl_envs.json')
parser.add_argument('--max_path_length', type=int, default=200)
parser.add_argument('--enable_train_eval', default=False, action='store_true')
parser.add_argument('--num_initial_steps', type=int, default=1000)
parser.add_argument('--unbounded_eval_hist', default=False, action='store_true')

# Context
parser.add_argument('--hiddens_conext', nargs='+', type=int, default=[20], help='indicates hidden size of context next')
parser.add_argument('--enable_context', default=True, action='store_true')
parser.add_argument('--only_concat_context', type=int, default=3, help=' use conext')
parser.add_argument('--num_tasks_sample', type=int, default=5)
parser.add_argument('--num_train_steps', type=int, default=400)
parser.add_argument('--min_buffer_size', type=int, default=100000, help='this indicates a condition to start using num_train_steps')
parser.add_argument('--history_length', type=int, default=30)

# Safe
parser.add_argument('--enable_CBF', default=True, action='store_true')
parser.add_argument('--enable_GP', default=True, action='store_true')
parser.add_argument('--CBF_name', default='ZCBF')
parser.add_argument('--CBF_penalty', type=int, default=20, help='the penalty factor for CBF-based compensation in R')

# Meta
parser.add_argument('--beta_clip', default=1.5, type=float, help='Range to clip beta term in CSC')
parser.add_argument('--snapshot_size', type=int, default=2000, help='Snapshot size for a task')
parser.add_argument('--prox_coef', default=0.1, type=float, help='Prox lambda')
parser.add_argument('--meta_batch_size', default=10, type=int, help='Meta batch size: number of sampled tasks per itr')
parser.add_argument('--enable_adaptation', default=True, action='store_true')
parser.add_argument('--main_snap_iter_nums', default=200, type=int, help='how many times adapt using train task but with csc')
parser.add_argument('--snap_iter_nums', default=100, type=int, help='how many times adapt using eval task')
parser.add_argument('--type_of_training', default='td3', help='td3')
parser.add_argument('--lam_csc', default=0.1, type=float, help='logisitc regression reg, smaller means stronger reg')
parser.add_argument('--use_ess_clipping', default=True, action='store_true')
parser.add_argument('--enable_beta_obs_cxt', default=True, action='store_true', help='if true concat obs + context')
parser.add_argument('--sampling_style', default='replay', help='replay')
parser.add_argument('--sample_mult', type=int, default=5, help='sample multipler of main_iter for adapt method')
parser.add_argument('--use_epi_len_steps', default=True, action='store_true')
parser.add_argument('--use_normalized_beta', default=False, action='store_true', help='normalized beta_score')
parser.add_argument('--reset_optims', default=False, action='store_true', help='init optimizers at the start of adaptation')
parser.add_argument('--lr_milestone', default=-1, type=int, help='reduce learning rate after this epoch')
parser.add_argument('--lr_gamma', default=0.8, type=float, help='learning rate decay')

# test
parser.add_argument('--test', default=False, action='store_true', help='test for one task')
parser.add_argument('--multi_test', default=False, action='store_true', help='test for multi-tasks')
parser.add_argument('--max_test_episode', default=4, type=int, help='test epiosde')
parser.add_argument('--test_tasks', default=50, type=int, help='the number of test tasks')


def update_lr(eparams, iter_num, alg_mth):
    # reduce learning rate after the setting milestone epoch
    if iter_num > eparams.lr_milestone:
        new_lr = eparams.lr * eparams.lr_gamma

        for param_group in alg_mth.actor_optimizer.param_groups:
            param_group['lr'] = new_lr

        for param_group in alg_mth.critic_optimizer.param_groups:
            param_group['lr'] = new_lr


def take_snapshot(args, fname, model, update):
    # save the current model and some other info
    fname_ck = fname + '.pt'
    fname_json = fname + '.json'
    curr_state_actor = get_state(model.actor)
    curr_state_critic = get_state(model.critic)

    print('Saving a checkpoint for iteration %d in %s' % (update, fname_ck))
    checkpoint = {
        'args': args.__dict__,
        'model_states_actor': curr_state_actor,
        'model_states_critic': curr_state_critic,
    }
    torch.save(checkpoint, fname_ck)

    del checkpoint['model_states_actor']
    del checkpoint['model_states_critic']
    del curr_state_actor
    del curr_state_critic

    dump_to_json(fname_json, checkpoint)


def setup_log_and_checkpoints(args):
    # create folder if not there
    create_dir(args.check_point_dir)

    fname = str.lower(args.env_name) + '_' + args.alg_name + '_' + time.strftime('%H-%M-%S')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    fname_log = os.path.join(args.log_dir, fname)
    fname_eval = os.path.join(fname_log, 'eval.csv')
    fname_adapt = os.path.join(fname_log, 'adapt.csv')
    return os.path.join(args.check_point_dir, fname), fname_log, fname_eval, fname_adapt


def make_env(eparams):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # this is based on PEARL paper that fixes set of sampels
    from misc.env_meta import build_PEARL_envs
    env = build_PEARL_envs(
        seed=eparams.seed,
        env_name=eparams.env_name,
        params=eparams,
    )
    return env


def sample_env_tasks(env, eparams):
    tasks = env.get_all_task_idx()
    train_tasks = list(tasks[:eparams.n_train_tasks])
    eval_tasks = list(tasks[-eparams.n_eval_tasks:])
    return train_tasks, eval_tasks


def config_tasks_envs(eparams):
    configs = read_json(eparams.env_configs)[eparams.env_name]
    temp_params = vars(eparams)
    for k, v in configs.items():
        temp_params[k] = v


def evaluate_policy(eval_env,
                    policy,
                    eps_num,
                    itr,
                    etasks,
                    eparams,
                    meta_learner=None,
                    train_tasks_buffer=None,
                    train_replay_buffer=None,
                    dynamics_gp=None,
                    cbf=None,
                    msg='Evaluation'):
    if eparams.unbounded_eval_hist:  # increase seq length to max_path_length
        eval_hist_len = eparams.max_path_length
        print('Eval uses unbounded_eval_hist of length: ', eval_hist_len)
    else:
        eval_hist_len = eparams.history_length
        print('Eval uses history of length: ', eval_hist_len)

    # adaptation step
    if meta_learner and eparams.enable_adaptation:
        meta_learner.save_model_states()

    # return infos
    all_task_rewards = []
    all_task_nadir = []
    all_task_action = []
    all_task_unsafe = []
    all_task_stable = []
    all_task_safe_stable = []
    all_task_worst = []
    dc_rewards = []

    for tidx in etasks:
        if args.multi_test:
            eval_env.set_task(tidx)
        else:
            eval_env.reset_task(tidx)

        # adaptation step #
        if meta_learner and eparams.enable_adaptation:
            eval_task_buffer, avg_data_collection = collect_data_for_adaptaion(eval_env, policy, tidx, eparams)
            stats_main, stats_csv = meta_learner.adapt(train_replay_buffer=train_replay_buffer,
                                                       train_tasks_buffer=train_tasks_buffer,
                                                       eval_task_buffer=eval_task_buffer,
                                                       task_id=tidx,
                                                       snap_iter_nums=eparams.snap_iter_nums,
                                                       main_snap_iter_nums=eparams.main_snap_iter_nums,
                                                       sampling_style=eparams.sampling_style,
                                                       sample_mult=eparams.sample_mult,
                                                       cbf=cbf,
                                                       dynamics_gp=dynamics_gp
                                                       )
            dc_rewards.append(avg_data_collection)
            print('--------Adaptation-----------')
            print('Task: ', tidx)
            print("Use test task's buffer")
            print(f"critic loss:{stats_csv['critic_loss']} | actor loss:{stats_csv['actor_loss']}")
            print("Use train tasks' buffers")
            print(f"critic loss:{stats_main['prox_critic']} | actor loss:{stats_main['prox_actor']}")
            print(f"avg_prox_coef:{stats_main['avg_prox_coef']} | beta_score:{stats_main['beta_score']}")
            print('-----------------------------')

        # csv infos
        avg_reward = 0
        avg_nadir = 0
        min_nadir = 0
        avg_action = 0
        avg_stable_t = 0
        sum_unsafe_num = 0
        sum_safe_stable_num = 0
        if args.enable_CBF:
            dynamics_gp.firstIter = 1

        if args.multi_test:
            eparams.num_evals = 2

        for eval_t in range(eparams.num_evals):
            obs = eval_env.reset()
            done = False
            step = 0

            # history
            rewards_hist = deque(maxlen=eval_hist_len)
            actions_hist = deque(maxlen=eval_hist_len)
            obsvs_hist = deque(maxlen=eval_hist_len)

            zero_action = np.zeros(eval_env.action_space.shape[0])
            zero_obs = np.zeros(obs.shape)
            for _ in range(eval_hist_len):
                rewards_hist.append(0)
                actions_hist.append(zero_action.copy())
                obsvs_hist.append(zero_obs.copy())

            rand_acttion = np.random.normal(0, eparams.expl_noise, size=eval_env.action_space.shape[0])
            rand_acttion = rand_acttion.clip(eval_env.action_space.low, eval_env.action_space.high)
            rewards_hist.append(0)
            actions_hist.append(rand_acttion.copy())
            obsvs_hist.append(obs.copy())

            if args.enable_CBF:
                ep_obs, ep_action, ep_rewards, ep_action_bar = [], [], [], []
                ep_gamma_list = []
                nadir = [1]
                if eval_t > 1:
                    # only update GP model for 1 episode
                    args.enable_GP = False
            else:
                ep_obs, ep_action, ep_rewards = [], [], []
                ep_gamma_list = []
                nadir = [1]
            stable_t = 0
            while not done and step < eparams.max_path_length:
                np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten()
                np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)
                np_pre_obsvs = np.asarray(obsvs_hist, dtype=np.float32).flatten()
                action = policy.select_action(np.array(obs), np.array(np_pre_actions), np.array(np_pre_rewards),
                                              np.array(np_pre_obsvs))
                if args.enable_CBF:
                    action_rl = action[0] + [0]
                    # Utilize safety barrier function
                    if dynamics_gp.firstIter == 1:
                        [_, _, x], [f_, g_] = dynamics_gp.get_dynamics(obs, action_rl[0])
                        std = [0, 0]
                    else:
                        [_, _, x, std], [f_, g_] = dynamics_gp.get_GP_dynamics(obs, action_rl[0])
                    u_bar_, cur_gamma = cbf.control_barrier(args.CBF_name, action_rl[0], f_, g_, x, std)
                    action = action_rl + u_bar_

                    ep_action_bar.append(u_bar_)
                    ep_gamma_list.append(cur_gamma)
                else:
                    ep_gamma_list.append(0)
                new_obs, reward, done, infos = eval_env.step(action)
                if new_obs[0] < -0.81:
                    sum_unsafe_num += 1
                if infos['stable']:
                    stable_t += 1
                    if new_obs[0] > -0.51:
                        sum_safe_stable_num += 1
                if args.enable_CBF:
                    reward -= args.CBF_penalty * abs(u_bar_[0])
                avg_reward += reward
                step += 1

                # new becomes old
                rewards_hist.append(reward)
                if args.enable_CBF:
                    actions_hist.append(action_rl.copy())
                else:
                    actions_hist.append(action.copy())
                obsvs_hist.append(obs.copy())

                obs = new_obs.copy()
                if args.enable_CBF:
                    ep_obs.append(obs)
                    ep_rewards.append(reward)
                    ep_action_bar.append(u_bar_)
                    ep_action.append(action)
                    nadir.append(min(min(nadir), obs[0]))
                else:
                    ep_obs.append(obs)
                    ep_rewards.append(reward)
                    ep_action.append(action)
                    nadir.append(min(min(nadir), obs[0]))
            avg_stable_t += stable_t
            avg_nadir += min(nadir)
            min_nadir = min(nadir)
            avg_action += np.mean(np.abs(ep_action))

        avg_reward /= eparams.num_evals
        avg_nadir /= eparams.num_evals
        avg_action /= eparams.num_evals
        avg_stable_t /= eparams.num_evals

        all_task_rewards.append(avg_reward)
        all_task_nadir.append(avg_nadir)
        all_task_action.append(avg_action)
        all_task_stable.append(avg_stable_t * eparams.num_evals)

        all_task_unsafe.append(sum_unsafe_num)
        all_task_safe_stable.append(sum_safe_stable_num)
        all_task_worst.append(min_nadir)

        if args.enable_CBF:
            print(f"evaluation: [delta_PL:{eval_env.delta_Pl}, Rg:{eval_env.Rg}, Hc:{eval_env.Hc}] | nadir: {round(min(nadir), 4)} | mean action:{np.round(np.mean(np.abs(ep_action)), 4)} | ep_rewards:{np.round(np.sum(ep_rewards), 4)} | mean CBF:{np.round(np.mean(ep_action_bar), 4)} | stable_t:{stable_t} | mean_gamma:{round(sum(ep_gamma_list) / len(ep_gamma_list), 4)}")
            if args.enable_GP:
                path = {"Observation": np.concatenate(ep_obs).reshape((len(ep_obs), 2)),
                        "Action": np.concatenate(ep_action),
                        "Action_bar": np.concatenate(ep_action_bar),
                        "Reward": np.asarray(ep_rewards),
                        "Nadir": np.asarray(nadir)}
                dynamics_gp.update_GP_dynamics(path)
        else:
            print(f"evaluation: [delta_PL:{eval_env.delta_Pl}, Rg:{eval_env.Rg}, Hc:{eval_env.Hc}] | nadir: {round(min(nadir), 4)} | mean action:{np.round(np.mean(np.abs(ep_action)), 4)} | ep_rewards:{np.round(np.sum(ep_rewards), 4)} | stable_t:{stable_t}")

        # adaptation step
        # Roll-back
        if meta_learner and eparams.enable_adaptation:
            meta_learner.rollback()

            # add adapt data to a csv file
            log_data_adp = {}
            for k, v in stats_csv.items():
                if k in eparams.adapt_csv_hearder:
                    log_data_adp[k] = stats_csv[k]

            log_data_adp['csc_samples_neg'] = stats_csv['csc_info'][0]
            log_data_adp['csc_samples_pos'] = stats_csv['csc_info'][1]
            log_data_adp['train_acc'] = stats_csv['csc_info'][2]
            log_data_adp['avg_rewards'] = avg_reward
            log_data_adp['one_raw_reward'] = avg_data_collection
            log_data_adp['tidx'] = tidx
            log_data_adp['eps_num'] = eps_num
            log_data_adp['iter'] = itr
            log_data_adp['ep_nadir'] = min(nadir)
            log_data_adp['ep_gamma'] = sum(ep_gamma_list) / len(ep_gamma_list)
            log_data_adp['mean_ep_action'] = np.mean(np.abs(ep_action))

            for k in stats_main.keys():
                if k in eparams.adapt_csv_hearder:
                    log_data_adp['main_' + k] = stats_main[k]
                elif 'main_' + k in eparams.adapt_csv_hearder:
                    log_data_adp['main_' + k] = stats_main[k]

            adapt_csv_stats.write(log_data_adp)

        # add test data to test.csv file
        if args.multi_test:
            log_data_test = {}
            log_data_test['test_task'] = tidx
            log_data_test['ep_nadir'] = min(nadir)
            log_data_test['avg_gamma'] = sum(ep_gamma_list) / len(ep_gamma_list)
            log_data_test['avg_rewards'] = avg_reward
            log_data_test['mean_ep_action'] = np.mean(np.abs(ep_action))
            log_data_test['avg_stable_t'] = avg_stable_t
            log_data_test['sum_safe_stable'] = sum_safe_stable_num
            log_data_test['sum_unsafe'] = sum_unsafe_num
            test_csv_stats.write(log_data_test)

    if meta_learner and eparams.enable_adaptation:
        msg += ' *** with Adapation *** '
        print('Avg rewards (only one eval loop) for all tasks before adaptation ', np.mean(dc_rewards))

    info = {}
    info['eval_reward'] = np.mean(all_task_rewards)
    info['eval_nadir'] = np.mean(all_task_nadir)
    info['eval_action'] = np.mean(all_task_action)
    info['eval_unsafe'] = np.sum(all_task_unsafe)
    info['eval_safe_stable'] = np.sum(all_task_safe_stable)
    info['eval_stable'] = np.sum(all_task_stable)
    info['eval_worst_nadir'] = np.min(all_task_worst)
    return info


def collect_data_for_adaptaion(eval_env, policy, tidx, eparams):
    # Step 0: Create eval buffers
    eval_task_buffer = MultiTasksSnapshot(max_size=args.snapshot_size)
    eval_task_buffer.init([tidx])
    # Step 1: Define some vars
    step = 0
    avg_reward = 0
    obs = eval_env.reset()
    done = False
    nadir = 1

    rewards_hist = deque(maxlen=eparams.history_length)
    actions_hist = deque(maxlen=eparams.history_length)
    obsvs_hist = deque(maxlen=eparams.history_length)

    next_hrews = deque(maxlen=eparams.history_length)
    next_hacts = deque(maxlen=eparams.history_length)
    next_hobvs = deque(maxlen=eparams.history_length)

    zero_action = np.zeros(eval_env.action_space.shape[0])
    zero_obs = np.zeros(obs.shape)

    for _ in range(eparams.history_length):
        rewards_hist.append(0)
        actions_hist.append(zero_action.copy())
        obsvs_hist.append(zero_obs.copy())

        # same thing for next_h*
        next_hrews.append(0)
        next_hacts.append(zero_action.copy())
        next_hobvs.append(zero_obs.copy())

    rewards_hist.append(0)
    obsvs_hist.append(obs.copy())

    rand_action = np.random.normal(0, eparams.expl_noise, size=eval_env.action_space.shape[0])
    rand_action = rand_action.clip(eval_env.action_space.low, eval_env.action_space.high)
    actions_hist.append(rand_action.copy())

    while not done and step < eparams.max_path_length:
        np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten()
        np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)
        np_pre_obsvs = np.asarray(obsvs_hist, dtype=np.float32).flatten()
        action = policy.select_action(np.array(obs), np.array(np_pre_actions), np.array(np_pre_rewards),
                                      np.array(np_pre_obsvs))
        if args.enable_CBF:
            action_rl = action[0] + [0]
            # Utilize safety barrier function
            if dynamics_gp.firstIter == 1:
                [_, _, x], [f_, g_] = dynamics_gp.get_dynamics(obs, action_rl[0])
                std = [0, 0]
            else:
                [_, _, x, std], [f_, g_] = dynamics_gp.get_GP_dynamics_prev(obs, action_rl[0])
            u_bar_, _ = cbf.control_barrier("ZCBF", action_rl[0], f_, g_, x, std)
            action = action_rl + u_bar_
        new_obs, reward, done, _ = eval_env.step(action)
        if args.enable_CBF:
            reward -= args.CBF_penalty * abs(u_bar_[0])
        avg_reward += reward

        if step + 1 == args.max_path_length:
            done_bool = 0
        else:
            done_bool = float(done)

        next_hrews.append(reward)
        if args.enable_CBF:
            next_hacts.append(action_rl.copy())
        else:
            next_hacts.append(action.copy())
        next_hobvs.append(obs.copy())

        # np_next_hacts and np_next_hrews are required for TD3 alg
        np_next_hacts = np.asarray(next_hacts, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
        np_next_hrews = np.asarray(next_hrews, dtype=np.float32)  # (hist, )
        np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()  # (hist, )
        if args.enable_CBF:
            eval_task_buffer.add(tidx, (obs, new_obs, action_rl, reward, done_bool,
                                        np_pre_actions, np_pre_rewards, np_pre_obsvs,
                                        np_next_hacts, np_next_hrews, np_next_hobvs))
        else:
            eval_task_buffer.add(tidx, (obs, new_obs, action, reward, done_bool,
                                        np_pre_actions, np_pre_rewards, np_pre_obsvs,
                                        np_next_hacts, np_next_hrews, np_next_hobvs))
        step += 1

        # new becomes old
        rewards_hist.append(reward)
        if args.enable_CBF:
            actions_hist.append(action_rl.copy())
            # actions_hist.append(action.copy())
        else:
            actions_hist.append(action.copy())
        obsvs_hist.append(obs.copy())

        obs = new_obs.copy()
        nadir = min(nadir, obs[0])

    return eval_task_buffer, avg_reward


def adjust_number_train_iters(buffer_size, num_train_steps, bsize, min_buffer_size,
                              episode_timesteps, use_epi_len_steps=False):

    # This adjusts number of gradient updates given sometimes there is not enough data in buffer
    if use_epi_len_steps and episode_timesteps > 1 and buffer_size < min_buffer_size:
        return episode_timesteps

    if buffer_size < num_train_steps or buffer_size < min_buffer_size:
        temp = int(buffer_size / bsize % num_train_steps) + 1

        if temp < num_train_steps:
            num_train_steps = temp

    return num_train_steps


def test_setup(load_path):
    max_action = float(env.action_space.high[0])
    if len(env.observation_space.shape) == 1:
        import models.networks as net

        # This part to add context network
        if args.enable_context:
            reward_dim = 1
            input_dim_context = env.action_space.shape[0] + reward_dim
            args.output_dim_conext = (env.action_space.shape[0] + reward_dim) * 2

            if args.only_concat_context == 3:  # means use LSTM with action_reward_state as an input
                input_dim_context = env.action_space.shape[0] + reward_dim + env.observation_space.shape[0]
                actor_idim = [env.observation_space.shape[0] + args.hiddens_conext[0]]
                args.output_dim_conext = args.hiddens_conext[0]
                dim_others = args.hiddens_conext[0]

            else:
                raise ValueError(" %d args.only_concat_context is not supported" % (args.only_concat_context))

        else:
            actor_idim = env.observation_space.shape
            dim_others = 0
            input_dim_context = None
            args.output_dim_conext = 0

        actor_net = net.Actor(action_space=env.action_space,
                              hidden_sizes=args.hidden_sizes,
                              input_dim=actor_idim,
                              max_action=max_action,
                              enable_context=args.enable_context,
                              hiddens_dim_conext=args.hiddens_conext,
                              input_dim_context=input_dim_context,
                              output_conext=args.output_dim_conext,
                              only_concat_context=args.only_concat_context,
                              history_length=args.history_length,
                              obsr_dim=env.observation_space.shape[0],
                              device=device
                              ).to(device)

        actor_target_net = net.Actor(action_space=env.action_space,
                                     hidden_sizes=args.hidden_sizes,
                                     input_dim=actor_idim,
                                     max_action=max_action,
                                     enable_context=args.enable_context,
                                     hiddens_dim_conext=args.hiddens_conext,
                                     input_dim_context=input_dim_context,
                                     output_conext=args.output_dim_conext,
                                     only_concat_context=args.only_concat_context,
                                     history_length=args.history_length,
                                     obsr_dim=env.observation_space.shape[0],
                                     device=device
                                     ).to(device)

        critic_net = net.Critic(action_space=env.action_space,
                                hidden_sizes=args.hidden_sizes,
                                input_dim=env.observation_space.shape,
                                enable_context=args.enable_context,
                                dim_others=dim_others,
                                hiddens_dim_conext=args.hiddens_conext,
                                input_dim_context=input_dim_context,
                                output_conext=args.output_dim_conext,
                                only_concat_context=args.only_concat_context,
                                history_length=args.history_length,
                                obsr_dim=env.observation_space.shape[0],
                                device=device
                                ).to(device)

        critic_target_net = net.Critic(action_space=env.action_space,
                                       hidden_sizes=args.hidden_sizes,
                                       input_dim=env.observation_space.shape,
                                       enable_context=args.enable_context,
                                       dim_others=dim_others,
                                       hiddens_dim_conext=args.hiddens_conext,
                                       input_dim_context=input_dim_context,
                                       output_conext=args.output_dim_conext,
                                       only_concat_context=args.only_concat_context,
                                       history_length=args.history_length,
                                       obsr_dim=env.observation_space.shape[0],
                                       device=device
                                       ).to(device)

    else:
        raise ValueError(
            "%s model is not supported for %s env" % (args.env_name, env.observation_space.shape))

    checkpoint = torch.load(load_path + '.pt')
    actor_net.load_state_dict(checkpoint['model_states_actor'])
    actor_target_net.load_state_dict(checkpoint['model_states_actor'])
    critic_net.load_state_dict(checkpoint['model_states_critic'])
    critic_target_net.load_state_dict(checkpoint['model_states_critic'])

    if str.lower(args.alg_name) == 'adapsafe':
        # tdm3 uses specific runner
        from misc.runner_multi_snapshot import Runner
        from algs.AdapSafe.multi_tasks_snapshot import MultiTasksSnapshot
        import algs.AdapSafe.adapsafe as alg

        alg = alg.AdapSafe(actor=actor_net,
                           actor_target=actor_target_net,
                           critic=critic_net,
                           critic_target=critic_target_net,
                           lr=args.lr,
                           gamma=args.gamma,
                           ptau=args.ptau,
                           policy_noise=args.policy_noise,
                           noise_clip=args.noise_clip,
                           policy_freq=args.policy_freq,
                           batch_size=args.batch_size,
                           max_action=max_action,
                           beta_clip=args.beta_clip,
                           prox_coef=args.prox_coef,
                           type_of_training=args.type_of_training,
                           lam_csc=args.lam_csc,
                           use_ess_clipping=args.use_ess_clipping,
                           enable_beta_obs_cxt=args.enable_beta_obs_cxt,
                           use_normalized_beta=args.use_normalized_beta,
                           reset_optims=args.reset_optims,
                           device=device,
                           )
    else:
        raise ValueError("%s alg is not supported" % args.alg_name)
    return alg


if __name__ == "__main__":

    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    print('Read Tasks/Env config params and Update args')
    config_tasks_envs(args)

    print(args.__dict__)

    # Generic setups
    CUDA_AVAL = torch.cuda.is_available()

    if not args.disable_cuda and CUDA_AVAL:
        gpu_id = "cuda:" + str(args.gpu_id)
        device = torch.device(gpu_id)
        print("**** We Use GPU %s ****" % gpu_id)

    else:
        device = torch.device('cpu')
        print("**** No GPU detected or GPU usage is disabled, sorry! ****")

    if not args.disable_cuda and CUDA_AVAL and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("****** cudnn.deterministic is set ******")

    ##############################
    # Init env, model, alg, batch generator etc
    # Step 1: Build env
    # Step 2: Build model
    # Step 3: Initiate Alg e.g. a2c
    # Step 4: Initiate batch/rollout generator
    ##############################

    # Step 1: Build env #
    env = make_env(args)
    # SEED
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timesteps_since_eval = 0
    episode_num = 0
    update_iter = 0
    sampling_loop = 0

    epinfobuf = deque(maxlen=args.n_train_tasks)
    epinfobuf_v2 = deque(maxlen=args.n_train_tasks)

    # (MULTI-)TEST #
    if args.test or args.multi_test:

        methods_list = ['AdapSafe']

        # save fig
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.use('Agg')
        plt.figure()
        fig, ax = plt.subplots(2, 4, figsize=(15, 8))

        for method in methods_list:
            if method == 'TD3':
                load_path = "./ck/TEST/TD3/freq_control_mql_10-23-01"
                print_name = 'TD3'
                alpha = 0
                env.action_bound = [-1.5, 1.5]
                args.enable_CBF = False
            elif method == 'MQL':
                load_path = "./ck/TEST/MQL/freq_control_mql_04-30-50"
                print_name = 'MQL'
                alpha = 0
                env.action_bound = [-1.5, 1.5]
                args.enable_CBF = False
            elif method == 'CBF-TD3':
                load_path = "./ck/TEST/CBF/freq_control_mql_08-55-27"
                print_name = 'CBF-TD3'
                env.action_bound = [-1.5, 1.5]
                alpha = [1, 1, 1, 1]
                args.enable_CBF = True
            else:
                load_path = "./ck/TEST/AdapSafe/freq_control_mql_22-43-15"
                print_name = 'AdapSafe'
                env.action_bound = [-1.5, 1.5]
                alpha = [1, 1, 0.1, 0.5]
                args.enable_CBF = True
            configs = read_json(load_path + '.json')['args']
            temp_params = vars(args)
            for k, v in configs.items():
                temp_params[k] = v
            args.test = True
            args.max_test_episode = 2
            alg = test_setup(load_path)

            if args.enable_CBF:
                dynamics_gp = DynamicGP(env)
                cbf = CBFController(env)
                cbf.build_barrier()
                dynamics_gp.build_GP_model()
                dynamics_gp.firstIter = 1

            # TEST #
            if not args.multi_test:
                cur_time = time.strftime('%m-%d-%H-%M-%S')
                if not os.path.isdir(f"./TEST_DATA/{cur_time}"):
                    os.mkdir(f"./TEST_DATA/{cur_time}")

                # we set two envs for (single) test
                # when run t_change ep, we change env from old to new
                t_change = 1
                old_hc, old_rg, old_pl = 4, 0.3, 1.8
                new_hc, new_rg, new_pl = 4, 0.24, 2.7

                for t in range(int(args.max_test_episode)):
                    if t >= t_change:
                        test_task = {'Hc': new_hc, 'Rg': new_rg, 'PL': new_pl}
                        update_gp_symbol = t - t_change
                    else:
                        test_task = {'Hc': old_hc, 'Rg': old_rg, 'PL': old_pl}
                        update_gp_symbol = t
                    print('-----------------------------')
                    print("Test Alg:", method)
                    print("Name of env:", args.env_name)
                    print("Observation_space:", env.observation_space)
                    print("Action space:", env.action_space)
                    print("Test tasks:", test_task)
                    print('----------------------------')
                    env.set_task(test_task)

                    step = 0
                    avg_reward = 0
                    prev_reward = 0
                    obs = env.reset()
                    done = False

                    test_obs = []
                    test_action = []
                    ep_rewards = 0

                    rewards_hist = deque(maxlen=args.history_length)
                    actions_hist = deque(maxlen=args.history_length)
                    obsvs_hist = deque(maxlen=args.history_length)

                    next_hrews = deque(maxlen=args.history_length)
                    next_hacts = deque(maxlen=args.history_length)
                    next_hobvs = deque(maxlen=args.history_length)

                    zero_action = np.zeros(env.action_space.shape[0])
                    zero_obs = np.zeros(obs.shape)
                    for _ in range(args.history_length):
                        rewards_hist.append(0)
                        actions_hist.append(zero_action.copy())
                        obsvs_hist.append(zero_obs.copy())

                        # same thing for next_h*
                        next_hrews.append(0)
                        next_hacts.append(zero_action.copy())
                        next_hobvs.append(zero_obs.copy())

                    rewards_hist.append(0)
                    obsvs_hist.append(obs.copy())

                    rand_action = np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])
                    rand_action = rand_action.clip(env.action_space.low, env.action_space.high)
                    actions_hist.append(rand_action.copy())

                    path = {}
                    stable_t = 0
                    while not done and step < args.max_path_length:
                        np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten()
                        np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)
                        np_pre_obsvs = np.asarray(obsvs_hist, dtype=np.float32).flatten()
                        action = alg.select_action(np.array(obs), np.array(np_pre_actions), np.array(np_pre_rewards), np.array(np_pre_obsvs))
                        pre_action = action.copy()
                        if args.enable_CBF:
                            if dynamics_gp.firstIter == 1:
                                [f, g, x], [f_, g_] = dynamics_gp.get_dynamics(obs, action[0])
                                std = [0, 0]
                            else:
                                [f, g, x, std], [f_, g_] = dynamics_gp.get_GP_dynamics(np.array(obs), action)
                            u_bar_, gamma = cbf.control_barrier_test(args.CBF_name, alpha, action, f_, g_, x, std)
                            action += u_bar_
                        new_obs, reward, done, infos = env.step(action)
                        if infos['stable']:
                            stable_t += 1
                        if args.enable_CBF:
                            reward -= args.CBF_penalty * abs(u_bar_[0])
                        if args.enable_CBF:
                            print(f"step:{step} | delta_f:{obs[0]} | action:{action} | CBF:{u_bar_} | reward:{reward} |gamma:{gamma} | stable_t:{stable_t}")
                        else:
                            print(f"step:{step} | delta_f:{obs[0]} | action:{action} | reward:{reward} | stable_t:{stable_t}")
                        test_obs.append(np.array(new_obs))
                        test_action.append(action[0])
                        ep_rewards += reward

                        avg_reward += reward
                        if args.enable_CBF:
                            if (step + 1) % 10 == 0 and update_gp_symbol <= 0:
                                path["Observation"] = np.array(test_obs[step - 9:step + 1])
                                path["Action"] = np.array(test_action[step - 9:step + 1])
                                dynamics_gp.update_GP_dynamics(path)
                                dynamics_gp.firstIter = 0
                        if step + 1 == args.max_path_length:
                            done_bool = 0
                        else:
                            done_bool = float(done)

                        if args.enable_CBF:
                            next_hrews.append(reward)
                            next_hacts.append(pre_action.copy())
                            next_hobvs.append(obs.copy())

                            step += 1

                            # new becomes old
                            rewards_hist.append(reward)
                            actions_hist.append(pre_action.copy())
                            obsvs_hist.append(obs.copy())
                        else:
                            next_hrews.append(reward)
                            next_hacts.append(action.copy())
                            next_hobvs.append(obs.copy())

                            step += 1

                            # new becomes old
                            rewards_hist.append(reward)
                            actions_hist.append(action.copy())
                            obsvs_hist.append(obs.copy())

                        obs = new_obs.copy()
                    print("ep rewards:", ep_rewards)
                    import pandas as pd

                    dataframe = pd.DataFrame({'delta_f': np.array(test_obs)[:, 0], 'action': test_action})
                    dataframe.to_csv(f"./TEST_DATA/{cur_time}/{method}_alpha_{alpha}_[{test_task['Hc']},{test_task['Rg']},{test_task['PL']}]_{t}.csv", index=False, sep=',')
            # MULTI-TEST #
            else:
                args.test_csv_hearder = dict.fromkeys(['test_task', 'ep_nadir',
                                                       'avg_gamma', 'avg_rewards',
                                                       'mean_ep_action', 'avg_stable_t',
                                                       'sum_safe_stable', 'sum_unsafe'])
                if not os.path.isdir(f"./MULTI_TEST_DATA/{time.strftime('%Y-%m-%d')}"):
                    os.mkdir(f"./MULTI_TEST_DATA/{time.strftime('%Y-%m-%d')}")
                fname_test = f"./MULTI_TEST_DATA/{time.strftime('%Y-%m-%d')}/{print_name}-{time.strftime('%H-%M-%S')}.csv"
                test_csv_stats = CSVWriter(fname_test, args.test_csv_hearder)
                np.random.seed(29)
                eval_tasks = env.sample_test_tasks(args.test_tasks)
                if args.enable_CBF:
                    args.enable_GP = True
                    eval_results = [evaluate_policy(env, alg, episode_num, update_iter, etasks=eval_tasks, eparams=args,dynamics_gp=dynamics_gp, cbf=cbf)]
                else:
                    eval_results = [evaluate_policy(env, alg, episode_num, update_iter, etasks=eval_tasks, eparams=args)]

    # TRAIN #
    else:
        # Step 2: Build model #
        create_dir(args.log_dir, cleanup=True)
        # create folder for save checkpoints
        ck_fname_part, log_file_dir, fname_csv_eval, fname_adapt = setup_log_and_checkpoints(args)
        logger.configure(dir=log_file_dir)
        wrt_csv_eval = None

        max_action = float(env.action_space.high[0])
        if len(env.observation_space.shape) == 1:
            import models.networks as net

            # add context network
            if args.enable_context:
                reward_dim = 1
                input_dim_context = env.action_space.shape[0] + reward_dim
                args.output_dim_conext = (env.action_space.shape[0] + reward_dim) * 2

                if args.only_concat_context == 3:  # means use LSTM with action_reward_state as an input
                    input_dim_context = env.action_space.shape[0] + reward_dim + env.observation_space.shape[0]
                    actor_idim = [env.observation_space.shape[0] + args.hiddens_conext[0]]
                    args.output_dim_conext = args.hiddens_conext[0]
                    dim_others = args.hiddens_conext[0]

                else:
                    raise ValueError(" %d args.only_concat_context is not supported" % (args.only_concat_context))

            else:
                actor_idim = env.observation_space.shape
                dim_others = 0
                input_dim_context = None
                args.output_dim_conext = 0

            actor_net = net.Actor(action_space=env.action_space,
                                  hidden_sizes=args.hidden_sizes,
                                  input_dim=actor_idim,
                                  max_action=max_action,
                                  enable_context=args.enable_context,
                                  hiddens_dim_conext=args.hiddens_conext,
                                  input_dim_context=input_dim_context,
                                  output_conext=args.output_dim_conext,
                                  only_concat_context=args.only_concat_context,
                                  history_length=args.history_length,
                                  obsr_dim=env.observation_space.shape[0],
                                  device=device
                                  ).to(device)

            actor_target_net = net.Actor(action_space=env.action_space,
                                         hidden_sizes=args.hidden_sizes,
                                         input_dim=actor_idim,
                                         max_action=max_action,
                                         enable_context=args.enable_context,
                                         hiddens_dim_conext=args.hiddens_conext,
                                         input_dim_context=input_dim_context,
                                         output_conext=args.output_dim_conext,
                                         only_concat_context=args.only_concat_context,
                                         history_length=args.history_length,
                                         obsr_dim=env.observation_space.shape[0],
                                         device=device
                                         ).to(device)

            critic_net = net.Critic(action_space=env.action_space,
                                    hidden_sizes=args.hidden_sizes,
                                    input_dim=env.observation_space.shape,
                                    enable_context=args.enable_context,
                                    dim_others=dim_others,
                                    hiddens_dim_conext=args.hiddens_conext,
                                    input_dim_context=input_dim_context,
                                    output_conext=args.output_dim_conext,
                                    only_concat_context=args.only_concat_context,
                                    history_length=args.history_length,
                                    obsr_dim=env.observation_space.shape[0],
                                    device=device
                                    ).to(device)

            critic_target_net = net.Critic(action_space=env.action_space,
                                           hidden_sizes=args.hidden_sizes,
                                           input_dim=env.observation_space.shape,
                                           enable_context=args.enable_context,
                                           dim_others=dim_others,
                                           hiddens_dim_conext=args.hiddens_conext,
                                           input_dim_context=input_dim_context,
                                           output_conext=args.output_dim_conext,
                                           only_concat_context=args.only_concat_context,
                                           history_length=args.history_length,
                                           obsr_dim=env.observation_space.shape[0],
                                           device=device
                                           ).to(device)

        else:
            raise ValueError("%s model is not supported for %s env" % (args.env_name, env.observation_space.shape))

        # Step 3: Initiate Alg e.g. TD3 #
        replay_buffer = Buffer(max_size=args.replay_size)
        if str.lower(args.alg_name) == 'adapsafe':
            # tdm3 uses specific runner
            from misc.runner_multi_snapshot import Runner
            from algs.AdapSafe.multi_tasks_snapshot import MultiTasksSnapshot
            import algs.AdapSafe.adapsafe as alg

            alg = alg.AdapSafe(actor=actor_net,
                               actor_target=actor_target_net,
                               critic=critic_net,
                               critic_target=critic_target_net,
                               lr=args.lr,
                               gamma=args.gamma,
                               ptau=args.ptau,
                               policy_noise=args.policy_noise,
                               noise_clip=args.noise_clip,
                               policy_freq=args.policy_freq,
                               batch_size=args.batch_size,
                               max_action=max_action,
                               beta_clip=args.beta_clip,
                               prox_coef=args.prox_coef,
                               type_of_training=args.type_of_training,
                               lam_csc=args.lam_csc,
                               use_ess_clipping=args.use_ess_clipping,
                               enable_beta_obs_cxt=args.enable_beta_obs_cxt,
                               use_normalized_beta=args.use_normalized_beta,
                               reset_optims=args.reset_optims,
                               device=device,
                               )

            # Step 4: Initiate batch/rollout generator #
            tasks_buffer = MultiTasksSnapshot(max_size=args.snapshot_size)
            if args.enable_CBF:
                dynamics_gp = DynamicGP(env)
                cbf = CBFController(env)
                cbf.build_barrier()
                dynamics_gp.build_GP_model()
                rollouts = Runner(env=env,
                                  model=alg,
                                  replay_buffer=replay_buffer,
                                  tasks_buffer=tasks_buffer,
                                  burn_in=args.burn_in,
                                  expl_noise=args.expl_noise,
                                  total_timesteps=args.total_timesteps,
                                  max_path_length=args.max_path_length,
                                  history_length=args.history_length,
                                  device=device,
                                  safe=True,
                                  cbf=cbf,
                                  cbf_name=args.CBF_name,
                                  cbf_penalty=args.CBF_penalty,
                                  gp=dynamics_gp,
                                  )
            else:
                rollouts = Runner(env=env,
                                  model=alg,
                                  replay_buffer=replay_buffer,
                                  tasks_buffer=tasks_buffer,
                                  burn_in=args.burn_in,
                                  expl_noise=args.expl_noise,
                                  total_timesteps=args.total_timesteps,
                                  max_path_length=args.max_path_length,
                                  history_length=args.history_length,
                                  device=device)

        else:
            raise ValueError("%s alg is not supported" % args.alg_name)

        print('-----------------------------')
        print("Name of env:", args.env_name)
        print("Observation_space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Task number:", args.n_tasks)
        print("Train tasks:", args.n_train_tasks)
        print("Eval tasks:", args.n_eval_tasks)
        print("######### Using Hist len %d #########" % (args.history_length))

        print("@@@@@@@@@ Using PEARL Env Setting @@@@@@@@@")
        print('----------------------------')

        if args.pre_train:
            path = "./ck/2022-06-30/freq_control_mql_15-12-02.pt"
            checkpoint = torch.load(path)
            actor_net.load_state_dict(checkpoint['model_states_actor'])
            actor_target_net.load_state_dict(checkpoint['model_states_actor'])
            critic_net.load_state_dict(checkpoint['model_states_critic'])
            critic_target_net.load_state_dict(checkpoint['model_states_critic'])

        model_dir = time.strftime('%Y-%m-%d')
        if not os.path.isdir(f"./ck/{model_dir}"):
            os.mkdir(f"./ck/{model_dir}")
        ck_fname_part = f"./ck/{model_dir}/" + str.lower(args.env_name) + '_' + args.alg_name + '_' + time.strftime("%H-%M-%S")

        # rollout/batch generator
        train_tasks, eval_tasks = sample_env_tasks(env, args)

        tasks_buffer.init(train_tasks)
        alg.set_tasks_list(train_tasks)

        take_snapshot(args, ck_fname_part, alg, 0)

        # Evaluate untrained policy
        if args.enable_CBF:
            args.enable_GP = True
            untrained_eval_infos = evaluate_policy(env, alg, episode_num, update_iter,
                                                   etasks=eval_tasks, eparams=args,
                                                   dynamics_gp=dynamics_gp, cbf=cbf)
            eval_results = [untrained_eval_infos['eval_reward']]
        else:
            args.enable_GP = True
            untrained_eval_infos = evaluate_policy(env, alg, episode_num, update_iter,
                                                   etasks=eval_tasks, eparams=args)
            eval_results = [untrained_eval_infos['eval_reward']]

        if args.enable_train_eval:
            train_subset = np.random.choice(train_tasks, len(eval_tasks))
            train_subset_tasks_eval_info = evaluate_policy(env, alg, episode_num, update_iter,
                                                           etasks=train_subset,
                                                           eparams=args,
                                                           msg='Train-Eval')
            train_subset_tasks_eval = train_subset_tasks_eval_info['eval_reward']
        else:
            train_subset_tasks_eval = 0

        wrt_csv_eval = CSVWriter(fname_csv_eval, {'total_timesteps': update_iter,
                                                  'eval_avg_rewards': eval_results[0],
                                                  'episode_num': episode_num,
                                                  'sampling_loop': sampling_loop,
                                                  'eval_avg_nadir': untrained_eval_infos['eval_nadir'],
                                                  'eval_avg_action': untrained_eval_infos['eval_action'],
                                                  'eval_sum_unsafe': untrained_eval_infos['eval_unsafe'],
                                                  'eval_sum_safe_stable': untrained_eval_infos['eval_safe_stable'],
                                                  'eval_sum_stable': untrained_eval_infos['eval_stable'],
                                                  'eval_worst_nadir': untrained_eval_infos['eval_worst_nadir'],
                                                  })
        wrt_csv_eval.write({'total_timesteps': update_iter,
                            'eval_avg_rewards': eval_results[0],
                            'episode_num': episode_num,
                            'sampling_loop': sampling_loop,
                            'eval_avg_nadir': untrained_eval_infos['eval_nadir'],
                            'eval_avg_action': untrained_eval_infos['eval_action'],
                            'eval_sum_unsafe': untrained_eval_infos['eval_unsafe'],
                            'eval_sum_safe_stable': untrained_eval_infos['eval_safe_stable'],
                            'eval_sum_stable': untrained_eval_infos['eval_stable'],
                            'eval_worst_nadir': untrained_eval_infos['eval_worst_nadir'],
                            })
        # keep track of adapt stats
        if args.enable_adaptation:
            args.adapt_csv_hearder = dict.fromkeys(['eps_num', 'iter', 'critic_loss', 'actor_loss',
                                                    'csc_samples_neg', 'csc_samples_pos', 'train_acc',
                                                    'snap_iter', 'beta_score', 'main_critic_loss',
                                                    'main_actor_loss', 'main_beta_score', 'main_prox_critic',
                                                    'main_prox_actor', 'main_avg_prox_coef',
                                                    'tidx', 'avg_rewards', 'one_raw_reward', 'ep_nadir', 'ep_gamma',
                                                    'mean_ep_action'])
            adapt_csv_stats = CSVWriter(fname_adapt, args.adapt_csv_hearder)

        # Start total timer
        tstart = time.time()

        # First fill up the replay buffer with all tasks
        max_cold_start = np.maximum(args.num_initial_steps * args.n_train_tasks, args.burn_in)
        print('Start burning for at least %d' % max_cold_start)
        keep_sampling = True
        avg_length = 0
        while keep_sampling:
            for idx in range(args.n_train_tasks):
                tidx = train_tasks[idx]
                env.reset_task(tidx)
                rollouts.paths = []
                data = rollouts.run(update_iter, keep_burning=True, task_id=tidx, early_leave=args.max_path_length)
                timesteps_since_eval += data['episode_timesteps']
                update_iter += data['episode_timesteps']
                epinfobuf.extend(data['epinfos'])
                epinfobuf_v2.extend(data['epinfos'])
                episode_num += 1
                avg_length += data['episode_timesteps']

                if update_iter >= max_cold_start:
                    keep_sampling = False
                    break

        print('There are %d samples in buffer now' % replay_buffer.size_rb())
        print('Average length %.2f for %d episode_nums for %d max_cold_start steps' % (
            avg_length / episode_num, episode_num, max_cold_start))
        print('Episode_nums/tasks %.2f and avg_len/tasks %.2f ' % (
            episode_num / args.n_train_tasks, avg_length / args.n_train_tasks))
        avg_epi_length = int(avg_length / episode_num)
        # already seen all tasks once
        sampling_loop = 1

        # Train and eval main loop
        train_iter = 0
        lr_updated = False
        args.enable_GP = False
        while update_iter < args.total_timesteps:
            # shuffle the ind
            train_indices = np.random.choice(train_tasks, len(train_tasks))
            for tidx in train_indices:

                # update learning rate
                if args.lr_milestone > -1 and not lr_updated and update_iter > args.lr_milestone:
                    update_lr(args, update_iter, alg)
                    lr_updated = True

                # run training to calculate loss, run backward, and update params
                stats_csv = None

                # adjust training steps
                adjusted_no_steps = adjust_number_train_iters(buffer_size=replay_buffer.size_rb(),
                                                              num_train_steps=args.num_train_steps,
                                                              bsize=args.batch_size,
                                                              min_buffer_size=args.min_buffer_size,
                                                              episode_timesteps=avg_epi_length,
                                                              use_epi_len_steps=args.use_epi_len_steps)

                if args.enable_CBF:
                    alg_stats, stats_csv = alg.train(replay_buffer=replay_buffer,
                                                     iterations=adjusted_no_steps,
                                                     tasks_buffer=tasks_buffer,
                                                     train_iter=train_iter,
                                                     task_id=tidx,
                                                     cbf=cbf,
                                                     dynamics_gp=dynamics_gp
                                                     )
                else:
                    alg_stats, stats_csv = alg.train(replay_buffer=replay_buffer,
                                                     iterations=adjusted_no_steps,
                                                     tasks_buffer=tasks_buffer,
                                                     train_iter=train_iter,
                                                     task_id=tidx,
                                                     )
                train_iter += 1
                # logging
                nseconds = time.time() - tstart
                # Calculate the fps (frame per second)
                fps = int(update_iter / nseconds)

                # progress.csv & log.txt文件
                if ((episode_num % args.log_interval == 0 or episode_num % len(
                        train_tasks) / 2 == 0) or episode_num == 1):
                    logger.record_tabular("nupdates", update_iter)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("total_timesteps", update_iter)
                    logger.record_tabular("critic_loss", float(alg_stats['critic_loss']))
                    logger.record_tabular("actor_loss", float(alg_stats['actor_loss']))
                    logger.record_tabular("episode_reward", float(data['episode_reward']))
                    logger.record_tabular('eprewmean', float(safemean([epinfo['r'] for epinfo in epinfobuf])))
                    logger.record_tabular('eplenmean', float(safemean([epinfo['l'] for epinfo in epinfobuf])))
                    logger.record_tabular("episode_num", episode_num)
                    logger.record_tabular("sampling_loop", sampling_loop)
                    logger.record_tabular("buffer_size", replay_buffer.size_rb())
                    logger.record_tabular("adjusted_no_steps", adjusted_no_steps)
                    logger.record_tabular("ep_nadir", float(data['ep_nadir']))
                    logger.record_tabular("ep_mean_action", float(data['ep_mean_action']))

                    if 'actor_mmd_loss' in alg_stats:
                        logger.record_tabular("critic_mmd_loss", float(alg_stats['critic_mmd_loss']))
                        logger.record_tabular("actor_mmd_loss", float(alg_stats['actor_mmd_loss']))

                    if 'beta_score' in alg_stats:
                        logger.record_tabular("beta_score", float(alg_stats['beta_score']))

                    logger.dump_tabular()
                    print(("Total T: %d Episode Num: %d Episode Len: %d Reward: %f") %
                          (update_iter, episode_num, data['episode_timesteps'], data['episode_reward']))

                    # print out some info about CSC
                    if stats_csv:
                        print(("CSC info:  critic_loss: %.4f actor_loss: %.4f No beta_score: %.4f ") %
                              (stats_csv['critic_loss'], stats_csv['actor_loss'], stats_csv['beta_score']))
                        if 'csc_info' in stats_csv:
                            print((
                                      "Number of examples used for CSC, prediction accuracy on train, and snap Iter: single: %d multiple tasks: %d  acc: %.4f snap_iter: %d ") %
                                  (stats_csv['csc_info'][0], stats_csv['csc_info'][1], stats_csv['csc_info'][2],
                                   stats_csv['snap_iter']))
                            print(("Prox info: prox_critic %.4f prox_actor: %.4f") % (
                                alg_stats['prox_critic'], alg_stats['prox_actor']))

                        if 'avg_prox_coef' in alg_stats and 'csc_info' in stats_csv:
                            print(("\ravg_prox_coef: %.4f" % (alg_stats['avg_prox_coef'])))

                # run eval
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq

                    if args.enable_adaptation:
                        if args.enable_CBF:
                            eval_info = evaluate_policy(env, alg, episode_num, update_iter,
                                                        etasks=eval_tasks, eparams=args,
                                                        meta_learner=alg,
                                                        train_tasks_buffer=tasks_buffer,
                                                        train_replay_buffer=replay_buffer,
                                                        dynamics_gp=dynamics_gp,
                                                        cbf=cbf
                                                        )
                        else:
                            eval_info = evaluate_policy(env, alg, episode_num, update_iter,
                                                        etasks=eval_tasks, eparams=args,
                                                        meta_learner=alg,
                                                        train_tasks_buffer=tasks_buffer,
                                                        train_replay_buffer=replay_buffer,
                                                        )

                    else:
                        if args.enable_CBF:
                            eval_info = evaluate_policy(env, alg, episode_num, update_iter,
                                                        etasks=eval_tasks, eparams=args,
                                                        dynamics_gp=dynamics_gp,
                                                        cbf=cbf
                                                        )
                        else:
                            eval_info = evaluate_policy(env, alg, episode_num, update_iter,
                                                        etasks=eval_tasks, eparams=args)

                    eval_results.append(eval_info['eval_reward'])

                    # Eval subset of train tasks
                    if args.enable_train_eval:

                        if args.enable_promp_envs == False:
                            train_subset = np.random.choice(train_tasks, len(eval_tasks))

                        else:
                            train_subset = None

                        train_subset_eval_info = evaluate_policy(env, alg, episode_num, update_iter,
                                                                 etasks=train_subset,
                                                                 eparams=args,
                                                                 msg='Train-Eval')
                        train_subset_tasks_eval = train_subset_eval_info['eval_reward']
                    else:
                        train_subset_tasks_eval = 0

                    # dump results
                    wrt_csv_eval.write({'total_timesteps': update_iter,
                                        'eval_avg_rewards': eval_results[0],
                                        'episode_num': episode_num,
                                        'sampling_loop': sampling_loop,
                                        'eval_avg_nadir': eval_info['eval_nadir'],
                                        'eval_avg_action': eval_info['eval_action'],
                                        'eval_sum_unsafe': eval_info['eval_unsafe'],
                                        'eval_sum_safe_stable': eval_info['eval_safe_stable'],
                                        'eval_sum_stable': eval_info['eval_stable'],
                                        'eval_worst_nadir': eval_info['eval_worst_nadir']})

                # save for every interval-th episode or for the last epoch
                if episode_num % args.save_freq == 0 or update_iter == args.total_timesteps:
                    model_dir = time.strftime('%Y-%m-%d')
                    if not os.path.isdir(f"./ck/{model_dir}"):
                        os.mkdir(f"./ck/{model_dir}")
                    ck_fname_part = f"./ck/{model_dir}/" + str.lower(
                        args.env_name) + '_' + args.alg_name + '_' + time.strftime("%H-%M-%S")
                    take_snapshot(args, ck_fname_part, alg, update_iter)

                # Interact and collect data until reset
                # should reset the queue, as new trail starts
                epinfobuf = deque(maxlen=args.n_train_tasks)
                avg_epi_length = 0
                if args.enable_CBF:
                    dynamics_gp.firstIter = 1
                rollouts.paths = []
                for sl in range(args.num_tasks_sample):

                    if sl > 0:
                        idx = np.random.randint(len(train_tasks))
                        tidx = train_tasks[idx]

                    if args.enable_promp_envs:
                        env.set_task(tidx)  # tidx for promp is task value
                    else:
                        env.reset_task(tidx)  # tidx here is an id

                    data = rollouts.run(update_iter, task_id=tidx)
                    timesteps_since_eval += data['episode_timesteps']
                    update_iter += data['episode_timesteps']
                    epinfobuf.extend(data['epinfos'])
                    epinfobuf_v2.extend(data['epinfos'])
                    episode_num += 1
                    avg_epi_length += data['episode_timesteps']

                avg_epi_length = int(avg_epi_length / args.num_tasks_sample)

            # just to keep track of how many times all training tasks have been seen
            sampling_loop += 1

        # Eval for the final time
        if args.enable_CBF:
            eval_info = evaluate_policy(env, alg, episode_num, update_iter, etasks=eval_tasks, eparams=args,
                                        dynamics_gp=dynamics_gp, cbf=cbf)
        else:
            eval_info = evaluate_policy(env, alg, episode_num, update_iter, etasks=eval_tasks, eparams=args)
        # Eval subset of train tasks
        if args.enable_promp_envs:
            train_subset = np.random.choice(train_tasks, len(eval_tasks))
        else:
            train_subset = None

        train_subset_tasks_eval_info = evaluate_policy(env, alg, episode_num, update_iter,
                                                       etasks=train_subset,
                                                       eparams=args,
                                                       dynamics_gp=dynamics_gp,
                                                       cbf=cbf,
                                                       msg='Train-Eval')
        train_subset_tasks_eval = train_subset_tasks_eval_info['eval_reward']

        eval_results.append(eval_info['eval_reward'])
        wrt_csv_eval.write({'total_timesteps': update_iter,
                            'eval_avg_rewards': eval_results[0],
                            'episode_num': episode_num,
                            'sampling_loop': sampling_loop,
                            'eval_avg_nadir': eval_info['eval_nadir'],
                            'eval_avg_action': eval_info['eval_action'],
                            'eval_sum_unsafe': eval_info['eval_unsafe'],
                            'eval_sum_safe_stable': eval_info['eval_safe_stable'],
                            'eval_sum_stable': eval_info['eval_stable'],
                            'eval_worst_nadir': eval_info['eval_worst_nadir']})
        wrt_csv_eval.close()
        print('Done')

