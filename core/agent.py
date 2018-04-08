import multiprocessing
from utils.replay_memory import NoLimitSequentialMemory
from utils.torch import *
from utils.tools import *
import math
import time


# ====================================================================================== #
# Agents for actor critic methods
# ====================================================================================== #
class ActorCriticAgent(object):
    def __init__(self, name, env_factory, policy, value, cfg, distinguish=None,
                 custom_reward=None, running_state=None, tensor_type=torch.DoubleTensor):
        self.id = cfg["env_name"] + "-" + name
        if distinguish:
            self.id = self.id + "-" + str(distinguish)
        self.env_factory = env_factory
        self.policy = policy
        self.value = value
        self.model_dict = {"policy": self.policy,
                           "value": self.value}
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.tensor = tensor_type

        self.tau = cfg["tau"]
        self.gamma = cfg["gamma"]
        self.mean_action = cfg["mean_action"] if "mean_action" in cfg else False
        self.num_threads = cfg["num_threads"] if "num_threads" in cfg else 1
        self.render = cfg["render"] if "render" in cfg else False
        self.gpu = cfg["gpu"] if "gpu" in cfg else False
        self.env_list = []
        for i in range(self.num_threads):
            self.env_list.append(self.env_factory(i))

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        if use_gpu and self.gpu:
            self.policy.cpu()

        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env_list[i + 1], self.policy, self.custom_reward, self.mean_action,
                           self.tensor, False, self.running_state, False, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action,
                                      self.tensor, self.render, self.running_state, True, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)

        batchs = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)

        if use_gpu and self.gpu:
            self.policy.cuda()

        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batchs.action), axis=0)
        log['action_min'] = np.min(np.vstack(batchs.action), axis=0)
        log['action_max'] = np.max(np.vstack(batchs.action), axis=0)
        return batchs, log

    def batch2tensor(self, batch):
        states = np_to_tensor(np.stack(batch.state))
        actions = np_to_tensor(np.stack(batch.action))
        rewards = np_to_tensor(np.stack(batch.reward))
        masks = np_to_tensor(np.stack(batch.mask).astype(np.float64))
        if use_gpu and self.gpu:
            states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
        values = self.value(Variable(states, volatile=True)).data

        # get advantage estimation from the trajectories
        advantages, value_targets = estimate_advantages(rewards, masks, values,
                                                        self.gamma, self.tau, use_gpu & self.gpu)

        batch = dict()
        batch["states"] = states
        batch["actions"] = actions
        batch["rewards"] = rewards
        batch["masks"] = masks
        batch["advantages"] = advantages
        batch["value_targets"] = value_targets
        return batch


# ====================================================================================== #
# Functions to collect samples
# ====================================================================================== #
def collect_samples(pid, queue, env, policy, custom_reward, mean_action,
                    tensor, render, running_state, update_rs, min_batch_size):
    torch.randn(pid, )
    log = dict()
    memory = NoLimitSequentialMemory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:
        state = env.reset()
        if running_state is not None:
            state = running_state(state, update=update_rs)
        reward_episode = 0
        t = 0

        for t in range(10000):
            state_var = Variable(tensor(state).unsqueeze(0), volatile=True)
            if mean_action:
                action = policy(state_var)[0].data[0].numpy()
            else:
                action = policy.select_action(state_var)[0].numpy()

            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state, update=update_rs)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            if done:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log
