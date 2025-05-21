import random
import numpy as np
import torch
import os
from garage.experiment import deterministic
from matplotlib import pyplot as plt
from matplotlib import animation
from time import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    deterministic.set_seed(seed)

def make_log_name(env_seq, args):
    if len(env_seq) == 1:
        env_name = env_seq[0]
        if '/' in env_name:
            env_name_list = env_name.split('/')
            env_name = env_name_list[0] + '_' + env_name_list[1]
        log_name = f'{args.env_type}_{args.rl_method}_{env_name}_{args.steps_per_task}_{args.seed}'
        if args.crelu:
            log_name = 'CReLU_' + log_name
        if args.wasserstein > 0:
            log_name = 'Wasserstein_{}_'.format(str(args.wasserstein)) + log_name
        return log_name

def simulate(env, policy, max_episode_length = 500, terminate_when_success = True):
    env.reset()
    obs = env.step(env.action_space.sample()).observation

    frames = []

    with torch.no_grad():
        for _ in range(max_episode_length - 1):
            frame = env.render(mode = 'human')
            frames.append(frame)
            action = policy.get_action(obs, 0)[0]
            es = env.step(action)
            if terminate_when_success: 
                if es.env_info.get("success", False): break
            obs = es.observation
    
    env.close()
    return frames

def save_animation(frames, path, fps = 10):
    tic = time()
    fig = plt.figure(figsize = (8, 8))
    a = frames[0]
    im = plt.imshow(
        a,
        interpolation = 'none',
        aspect = 'auto',
        vmin = 0,
        vmax = 1
    )
    plt.xticks([], [])
    plt.yticks([], [])

    def animate_func(i):
        im.set_array(frames[i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames = len(frames),
        interval = 1000 / fps
    )
    anim.save(path, fps = fps, extra_args = ['-vcodec', 'libx264'])
    toc = time()
    print("Animation saved to {} / Execution time: {:.2f}secs".format(path, toc - tic))

