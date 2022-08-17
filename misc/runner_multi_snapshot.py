import numpy as np
import torch
from collections import deque

class Runner:
    """
      This class generates batches of experiences
    """

    def __init__(self,
                 env,
                 model,
                 replay_buffer=None,
                 tasks_buffer=None,
                 burn_in=1e2,
                 expl_noise=0.1,
                 total_timesteps=1e6,
                 max_path_length=200,
                 history_length=1,
                 device='cpu',
                 safe=False,
                 cbf=None,
                 cbf_name=None,
                 cbf_penalty=None,
                 gp=None,
                 ):
        '''
            nsteps: number of steps
        '''
        self.model = model
        self.env = env
        self.burn_in = burn_in
        self.device = device
        self.episode_rewards = deque(maxlen=10)
        self.episode_lens = deque(maxlen=10)
        self.replay_buffer = replay_buffer
        self.expl_noise = expl_noise
        self.total_timesteps = total_timesteps
        self.max_path_length = max_path_length
        self.hist_len = history_length
        self.tasks_buffer = tasks_buffer
        self.enable_safe = safe
        self.gp = gp
        self.cbf = cbf
        self.cbf_name = cbf_name
        self.cbf_penalty = cbf_penalty
        self.barr_loss = 1
        self.paths = []

    def run(self, update_iter, keep_burning = False, task_id = None, early_leave = 200):
        '''
            This function add transition to replay buffer.
            Early_leave is used in just cold start to collect more data from various tasks,
            rather than focus on just few ones
        '''
        obs = self.env.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0
        uiter = 0
        reward_epinfos = []

        rewards_hist = deque(maxlen=self.hist_len)
        actions_hist = deque(maxlen=self.hist_len)
        obsvs_hist = deque(maxlen=self.hist_len)

        next_hrews = deque(maxlen=self.hist_len)
        next_hacts = deque(maxlen=self.hist_len)
        next_hobvs = deque(maxlen=self.hist_len)

        zero_action = np.zeros(self.env.action_space.shape[0])
        zero_obs = np.zeros(obs.shape)
        for _ in range(self.hist_len):
            rewards_hist.append(0)
            actions_hist.append(zero_action.copy())
            obsvs_hist.append(zero_obs.copy())

            # same thing for next_h*
            next_hrews.append(0)
            next_hacts.append(zero_action.copy())
            next_hobvs.append(zero_obs.copy())

        # now add obs to the seq
        rand_acttion = np.random.normal(0, self.expl_noise, size=self.env.action_space.shape[0])
        rand_acttion = rand_acttion.clip(self.env.action_space.low, self.env.action_space.high)
        rewards_hist.append(0)
        actions_hist.append(rand_acttion.copy())
        obsvs_hist.append(obs.copy())

        # Start collecting data
        if self.enable_safe:
            ep_obs, ep_action, ep_rewards, ep_action_bar, ep_action_BAR = [], [], [], [], []
            nadir = [1]
            if self.gp.firstIter == 1:
                pass
            else:
                self.gp.GP_model_prev = self.gp.GP_model.copy()
        else:
            ep_obs, ep_action, ep_rewards = [], [], []
            nadir = [1]

        stable_t = 0
        while not done and uiter < np.minimum(self.max_path_length, early_leave):

            # Convert actions_hist, rewards_hist to np.array and flatten them out
            # for example: hist =7, actin_dim = 11 --> np.asarray(actions_hist(7, 11)) ==> flatten ==> (77,)
            np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
            np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)  # (hist, )
            np_pre_obsers = np.asarray(obsvs_hist, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)

            # Select action randomly or according to policy
            if keep_burning or update_iter < self.burn_in:
                # action = self.env.action_space.sample()
                action = self.model.select_action(np.array(obs), np.array(np_pre_actions), np.array(np_pre_rewards), np.array(np_pre_obsers))
                if self.expl_noise != 0:
                    action = action + np.random.normal(0, self.expl_noise, size=self.env.action_space.shape[0])
                    action = action.clip(self.env.action_space.low, self.env.action_space.high)
                # print("burning action", action)
            else:
                action = self.model.select_action(np.array(obs), np.array(np_pre_actions), np.array(np_pre_rewards), np.array(np_pre_obsers))
                if self.expl_noise != 0:
                    action = action + np.random.normal(0, self.expl_noise, size=self.env.action_space.shape[0])
                    action = action.clip(self.env.action_space.low, self.env.action_space.high)
            if self.enable_safe:
                action_rl = action[0]
                action_rl = action_rl + [0]
                # Utilize safety barrier function
                if self.gp.firstIter == 1:
                    [_, _, x], [f_, g_] = self.gp.get_dynamics(obs, action_rl[0])
                    std = [0, 0]
                else:
                    [_, _, x, std], [f_, g_] = self.gp.get_GP_dynamics(np.array(obs), np.array(action_rl[0]))
                u_bar_, _ = self.cbf.control_barrier(self.cbf_name, action_rl[0], f_, g_, x, std)
                action = action_rl + u_bar_
            # Perform action
            new_obs, reward, done, infos = self.env.step(action)
            if infos['stable']:
                stable_t += 1
            if self.enable_safe:
                reward -= self.cbf_penalty * abs(u_bar_[0])
                # pass
            if episode_timesteps + 1 == self.max_path_length:
                done_bool = 0

            else:
                done_bool = float(done)

            episode_reward += reward
            reward_epinfos.append(reward)

            next_hrews.append(reward)
            if self.enable_safe:
                next_hacts.append(action_rl.copy())
            else:
                next_hacts.append(action.copy())
            next_hobvs.append(obs.copy())

            # np_next_hacts and np_next_hrews are required for TD3 alg
            np_next_hacts = np.asarray(next_hacts, dtype=np.float32).flatten()
            np_next_hrews = np.asarray(next_hrews, dtype=np.float32)
            np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()

            # Store data in replay buffer
            if self.enable_safe:
                self.replay_buffer.add((obs, new_obs, action_rl, reward, done_bool,
                                        np_pre_actions, np_pre_rewards, np_pre_obsers,
                                        np_next_hacts, np_next_hrews, np_next_hobvs))
            else:
                self.replay_buffer.add((obs, new_obs, action, reward, done_bool,
                                        np_pre_actions, np_pre_rewards, np_pre_obsers,
                                        np_next_hacts, np_next_hrews, np_next_hobvs))

            # This is snapshot buffer which has short memeory
            if self.enable_safe:
                self.tasks_buffer.add(task_id, (obs, new_obs, action_rl, reward, done_bool,
                                        np_pre_actions, np_pre_rewards, np_pre_obsers,
                                        np_next_hacts, np_next_hrews, np_next_hobvs))
            else:
                self.tasks_buffer.add(task_id, (obs, new_obs, action, reward, done_bool,
                                                np_pre_actions, np_pre_rewards, np_pre_obsers,
                                                np_next_hacts, np_next_hrews, np_next_hobvs))

            # new becomes old
            rewards_hist.append(reward)
            if self.enable_safe:
                actions_hist.append(action_rl.copy())
            else:
                actions_hist.append(action.copy())
            obsvs_hist.append(obs.copy())

            obs = new_obs.copy()
            episode_timesteps += 1
            update_iter += 1
            uiter += 1
            if self.enable_safe:
                ep_obs.append(np.array(obs))
                ep_rewards.append(reward)
                ep_action_bar.append(u_bar_)
                ep_action.append(action)
                nadir.append(min(min(nadir), obs[0]))
            else:
                ep_obs.append(np.array(obs))
                ep_rewards.append(reward)
                ep_action.append(action)
                nadir.append(min(min(nadir), obs[0]))

        if self.enable_safe:
            if keep_burning or update_iter < self.burn_in:
                path = {"Observation": np.concatenate(ep_obs).reshape((len(ep_obs), 2)),
                        "Action": np.concatenate(ep_action),
                        "Action_bar": np.concatenate(ep_action_bar),
                        "Reward": np.asarray(ep_rewards),
                        "Nadir": np.asarray(nadir),
                        "PL": np.array([self.env.delta_Pl] * len(ep_obs))}

                self.gp.update_GP_dynamics(path)
                self.paths.append(path)
            else:
                self.gp.firstIter = 0

        info = {}
        info['episode_timesteps'] = episode_timesteps
        info['update_iter'] = update_iter
        info['episode_reward'] = episode_reward
        info['epinfos'] = [{"r": round(sum(reward_epinfos), 6), "l": len(reward_epinfos)}]
        info['ep_nadir'] = min(nadir)
        info['ep_mean_action'] = np.mean(np.abs(ep_action))
        if self.enable_safe:
            print(f"sample: [delta_PL:{self.env.delta_Pl}, Rg:{self.env.Rg}, Hc:{self.env.Hc}] | steps:{update_iter} | burning:{keep_burning} | episode reward:{round(episode_reward, 4)} | nadir: {round(min(nadir), 4)} | mean action:{np.round(np.mean(np.abs(ep_action)), 4)} | mean CBF:{np.round(np.mean(np.abs(ep_action_bar)), 4)} | stable_t:{stable_t}")
        else:
            print(f"sample: [delta_PL:{self.env.delta_Pl}, Rg:{self.env.Rg}, Hc:{self.env.Hc}] | steps:{update_iter} | burning:{keep_burning} | episode reward:{round(episode_reward, 4)} | nadir: {round(min(nadir), 4)} | mean action:{np.round(np.mean(np.abs(ep_action)), 4)} | stable_t:{stable_t}")

        return info