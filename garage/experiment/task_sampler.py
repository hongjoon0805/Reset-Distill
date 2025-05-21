"""Efficient and general interfaces for sampling tasks for Meta-RL."""
# yapf: disable
import abc
import copy
import math
import gym
import deepmind_lab
from dm_control import suite

import numpy as np

from garage.envs import GymEnv, TaskNameWrapper, TaskOnehotWrapper, DeepmindLabEnv
from garage.envs.dm_control import DMControlEnv, dmc_task_name
from garage.sampler.env_update import (ExistingEnvUpdate, NewEnvUpdate,
                                       SetTaskUpdate)

# For Atari environment
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames

# yapf: enable


def _sample_indices(n_to_sample, n_available_tasks, with_replacement):
    """Select indices of tasks to sample.

    Args:
        n_to_sample (int): Number of environments to sample. May be greater
            than n_available_tasks.
        n_available_tasks (int): Number of available tasks. Task indices will
            be selected in the range [0, n_available_tasks).
        with_replacement (bool): Whether tasks can repeat when sampled.
            Note that if more tasks are sampled than exist, then tasks may
            repeat, but only after every environment has been included at
            least once in this batch. Ignored for continuous task spaces.

    Returns:
        np.ndarray[int]: Array of task indices.

    """
    if with_replacement:
        return np.random.randint(n_available_tasks, size=n_to_sample)
    else:
        blocks = []
        for _ in range(math.ceil(n_to_sample / n_available_tasks)):
            s = np.arange(n_available_tasks)
            np.random.shuffle(s)
            blocks.append(s)
        return np.concatenate(blocks)[:n_to_sample]


class TaskSampler(abc.ABC):
    """Class for sampling batches of tasks, represented as `~EnvUpdate`s.

    Attributes:
        n_tasks (int or None): Number of tasks, if known and finite.

    """

    @abc.abstractmethod
    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Note that if more tasks are sampled than exist, then tasks may
                repeat, but only after every environment has been included at
                least once in this batch. Ignored for continuous task spaces.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """

    @property
    def n_tasks(self):
        """int or None: The number of tasks if known and finite."""
        return None


class ConstructEnvsSampler(TaskSampler):
    """TaskSampler where each task has its own constructor.

    Generally, this is used when the different tasks are completely different
    environments.

    Args:
        env_constructors (list[Callable[Environment]]): Callables that produce
            environments (for example, environment types).

    """

    def __init__(self, env_constructors):
        self._env_constructors = env_constructors

    @property
    def n_tasks(self):
        """int: the number of tasks."""
        return len(self._env_constructors)

    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Note that if more tasks are sampled than exist, then tasks may
                repeat, but only after every environment has been included at
                least once in this batch. Ignored for continuous task spaces.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """
        return [
            NewEnvUpdate(self._env_constructors[i]) for i in _sample_indices(
                n_tasks, len(self._env_constructors), with_replacement)
        ]


class SetTaskSampler(TaskSampler):
    """TaskSampler where the environment can sample "task objects".

    This is used for environments that implement `sample_tasks` and `set_task`.
    For example, :py:class:`~HalfCheetahVelEnv`, as implemented in Garage.

    Args:
        env_constructor (type): Type of the environment.
        env (garage.Environment or None): Instance of env_constructor to sample
            from (will be constructed if not provided).
        wrapper (Callable[garage.Environment, garage.Environment] or None):
            Wrapper function to apply to environment.


    """

    def __init__(self, env_constructor, *, env=None, wrapper=None):
        self._env_constructor = env_constructor
        self._env = env or env_constructor()
        self._wrapper = wrapper

    @property
    def n_tasks(self):
        """int or None: The number of tasks if known and finite."""
        return getattr(self._env, 'num_tasks', None)

    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Note that if more tasks are sampled than exist, then tasks may
                repeat, but only after every environment has been included at
                least once in this batch. Ignored for continuous task spaces.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """
        return [
            SetTaskUpdate(self._env_constructor, task, self._wrapper)
            for task in self._env.sample_tasks(n_tasks)
        ]


class EnvPoolSampler(TaskSampler):
    """TaskSampler that samples from a finite pool of environments.

    This can be used with any environments, but is generally best when using
    in-process samplers with environments that are expensive to construct.

    Args:
        envs (list[Environment]): List of environments to use as a pool.

    """

    def __init__(self, envs):
        self._envs = envs

    @property
    def n_tasks(self):
        """int: the number of tasks."""
        return len(self._envs)

    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Since this cannot be easily implemented for an object pool,
                setting this to True results in ValueError.

        Raises:
            ValueError: If the number of requested tasks is larger than the
                pool, or with_replacement is set.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """
        if n_tasks > len(self._envs):
            raise ValueError('Cannot sample more environments than are '
                             'present in the pool. If more tasks are needed, '
                             'call grow_pool to copy random existing tasks.')
        if with_replacement:
            raise ValueError('EnvPoolSampler cannot meaningfully sample with '
                             'replacement.')
        envs = list(self._envs)
        np.random.shuffle(envs)
        return [ExistingEnvUpdate(env) for env in envs[:n_tasks]]

    def grow_pool(self, new_size):
        """Increase the size of the pool by copying random tasks in it.

        Note that this only copies the tasks already in the pool, and cannot
        create new original tasks in any way.

        Args:
            new_size (int): Size the pool should be after growning.

        """
        if new_size <= len(self._envs):
            return
        to_copy = _sample_indices(new_size - len(self._envs),
                                  len(self._envs),
                                  with_replacement=False)
        for idx in to_copy:
            self._envs.append(copy.deepcopy(self._envs[idx]))

class SuccessCounter(gym.Wrapper):
    """Helper class to keep count of successes in MetaWorld environments."""

    def __init__(self, env):
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action):
        es = self.env.step(action)
        
        if es.env_info.get("success", False):
            self.current_success = True
        if es.terminal or es.timeout:
            self.successes.append(self.current_success)
        return es

    def pop_successes(self):
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs):
        self.current_success = False
        return self.env.reset(**kwargs)

# env update에 맞게 고쳐야한다. 이것만 그렇게 고치면 sampler 고칠 필요 없이 바로 사용 가능할듯?
class ContinualLearningEnv(gym.Env):
    def __init__(self, envs, env_type, steps_per_env: int, seed: int, exploration_steps = 0):

        self.envs_before_make = envs
        self.env_type = env_type
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.exploration_steps = exploration_steps
        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self.cur_seq_idx = 0
        print('exploration stpes:', self.exploration_steps)

    def _check_steps_bound(self):
        if self.cur_step >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def pop_successes(self):
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes

    def step(self, action):
        self._check_steps_bound()
        es = self.envs[self.cur_seq_idx].step(action)

        # NOTE: This gives the task ID as part of the 'info' dict.
        es.env_info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        es.env_info["TimeLimit.truncated"] = False
        # print(self.cur_step)
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to be shorter than 200.
            es.env_info["TimeLimit.truncated"] = True

            print("Env changed: cur_step = {}, cur_seq_idx = {}".format(self.cur_step, self.cur_seq_idx))

            self.cur_step -= self.steps_per_env
            self.cur_seq_idx += 1
            
            self.steps_per_env += self.exploration_steps

        return es

    def reset(self):
        self._check_steps_bound()
        self.current_success = False
        return self.envs[self.cur_seq_idx].reset()

    def compute_reward(self):
        if self.env_type == 'metaworld':
            assert print('Metaworld do not need to compute substitute reward!')
        elif self.env_type == 'gym':
            return self.envs_before_make[self.cur_seq_idx].compute_reward
    
    def __call__(self):
        self.envs = []
        for env in self.envs_before_make:
            if self.env_type == 'metaworld':
                self.envs.append(env())
            else:
                env.reset()
                self.envs.append(env)


MW_TASKS_PER_ENV = 50


class CLTaskSampler(TaskSampler):
    """TaskSampler that distributes a Meta-World benchmark across workers.

    Args:
        benchmark (metaworld.Benchmark): Benchmark to sample tasks from.
        kind (str): Must be either 'test' or 'train'. Determines whether to
            sample training or test tasks from the Benchmark.
        wrapper (Callable[garage.Env, garage.Env] or None): Wrapper to apply to
            env instances.
        add_env_onehot (bool): If true, a one-hot representing the current
            environment name will be added to the environments. Should only be
            used with multi-task benchmarks.

    Raises:
        ValueError: If kind is not 'train' or 'test'. Also raisd if
            `add_env_onehot` is used on a metaworld meta learning (not
            multi-task) benchmark.

    """

    def __init__(self, benchmark, steps_per_env, seed, env_type='metaworld', wrapper=None, exploration_steps = 0):
        self._benchmark = benchmark
        self._steps_per_env = steps_per_env
        self.seed = seed
        self._env_type = env_type
        self._inner_wrapper = wrapper
        self._exploration_steps = exploration_steps
        
        
        self._task_indices = {}
        
        if self._env_type == 'metaworld':
            self._classes = benchmark.train_classes
            self._tasks = benchmark.train_tasks
            self._task_map = {
                env_name:
                [task for task in self._tasks if task.env_name == env_name]
                for env_name in self._classes.keys()
            }


        self.MT50_ENV_SEQS = [
            'assembly-v2','basketball-v2','bin-picking-v2','box-close-v2','button-press-topdown-v2',
            'button-press-topdown-wall-v2','button-press-v2','button-press-wall-v2','coffee-button-v2','coffee-pull-v2',
            'coffee-push-v2','dial-turn-v2','disassemble-v2','door-close-v2','door-lock-v2',
            'door-open-v2','door-unlock-v2','drawer-close-v2','drawer-open-v2','faucet-open-v2',
            'faucet-close-v2','hammer-v2','hand-insert-v2','handle-press-side-v2','handle-press-v2',
            'handle-pull-side-v2','handle-pull-v2','lever-pull-v2','peg-insert-side-v2','peg-unplug-side-v2',
            'pick-out-of-hole-v2','pick-place-v2','pick-place-wall-v2','plate-slide-back-side-v2','plate-slide-back-v2',
            'plate-slide-side-v2','plate-slide-v2','push-back-v2','push-v2','push-wall-v2',
            'reach-v2','reach-wall-v2','shelf-place-v2','soccer-v2','stick-pull-v2',
            'stick-push-v2','sweep-into-v2','sweep-v2','window-close-v2','window-open-v2',
        ]
        # Change GYM_ENV_SEQS in main_garage.py and self.GYM_ENV_SEQS in task_sampler.py to use dense reward.
        # self.GYM_ENV_SEQS = [
        #     'FetchReach-v1','FetchPush-v1','FetchSlide-v1','FetchPickAndPlace-v1',
        #     'HandReach-v0',
        #     'HandManipulateBlockRotateZ-v0','HandManipulateBlockRotateParallel-v0','HandManipulateBlockRotateXYZ-v0',
        #     'HandManipulateBlockFull-v0',
        #     'HandManipulateEggRotate-v0','HandManipulateEggFull-v0',
        #     'HandManipulatePenRotate-v0','HandManipulatePenFull-v0',
        # ]
        self.GYM_ENV_SEQS = [
            'FetchReach-v1','FetchPushDense-v1','FetchSlideDense-v1','FetchPickAndPlaceDense-v1',
            'HandReach-v0',
            'HandManipulateBlockRotateZ-v0','HandManipulateBlockRotateParallel-v0','HandManipulateBlockRotateXYZ-v0',
            'HandManipulateBlockFull-v0',
            'HandManipulateEggRotate-v0','HandManipulateEggFull-v0',
            'HandManipulatePenRotate-v0','HandManipulatePenFull-v0',
        ]

        self.DMLAB_ENV_SEQS = [
            'DeepmindLabSeekavoidArena01-v0',
            'DeepmindLabStairwayToMelon-v0',
            'DeepmindLabNavMazeStatic01-v0',
            'DeepmindLabNavMazeStatic02-v0',
            'DeepmindLabNavMazeStatic03-v0'
            'DeepmindLabNavMazeRandomGoal01-v0',
            'DeepmindLabNavMazeRandomGoal02-v0',
            'DeepmindLabNavMazeRandomGoal03-v0',
            'DeepmindLabLtChasm-v0',
            'DeepmindLabLtHallwaySlope-v0',
            'DeepmindLabLtHorseshoeColor-v0',
            'DeepmindLabLtSpaceBounceHard-v0',
        ]

        self.ATARI_ENV_SEQS = [
            'ALE/Adventure-v5', 	'ALE/AirRaid-v5', 			'ALE/Alien-v5', 		'ALE/Amidar-v5', 		'ALE/Assault-v5',       # 0,1,2,3,4
            'ALE/Asterix-v5', 		'ALE/Asteroids-v5', 		'ALE/Atlantis-v5',  	'ALE/BankHeist-v5', 	'ALE/BattleZone-v5',    # 5,6,7,8,9
            'ALE/BeamRider-v5', 	'ALE/Berzerk-v5', 			'ALE/Bowling-v5', 		'ALE/Boxing-v5', 		'ALE/Breakout-v5',      # 10,11,12,13,14
            'ALE/Carnival-v5', 		'ALE/Centipede-v5', 		'ALE/ChopperCommand-v5','ALE/CrazyClimber-v5',	'ALE/Defender-v5',      # 15,16,17,18,19
            'ALE/DemonAttack-v5', 	'ALE/DoubleDunk-v5', 		'ALE/ElevatorAction-v5','ALE/Enduro-v5', 		'ALE/FishingDerby-v5',  # 20,21,22,23,24
            'ALE/Freeway-v5', 		'ALE/Frostbite-v5', 		'ALE/Gopher-v5', 		'ALE/Gravitar-v5', 		'ALE/Hero-v5',          # 25,26,27,28,29
            'ALE/IceHockey-v5', 	'ALE/Jamesbond-v5', 		'ALE/JourneyEscape-v5', 'ALE/Kangaroo-v5', 		'ALE/Krull-v5',         # 30,31,32,33,34
            'ALE/KungFuMaster-v5', 	'ALE/MontezumaRevenge-v5',	'ALE/MsPacman-v5', 		'ALE/NameThisGame-v5', 	'ALE/Phoenix-v5',       # 35,36,37,38,39
            'ALE/Pitfall-v5', 		'ALE/Pong-v5', 				'ALE/Pooyan-v5', 		'ALE/PrivateEye-v5', 	'ALE/Qbert-v5',         # 40,41,42,43,44
            'ALE/Riverraid-v5', 	'ALE/RoadRunner-v5', 		'ALE/Robotank-v5', 		'ALE/Seaquest-v5', 		'ALE/Skiing-v5',        # 45,46,47,48,49
            'ALE/Solaris-v5', 		'ALE/SpaceInvaders-v5', 	'ALE/StarGunner-v5', 	'ALE/Tennis-v5', 		'ALE/TimePilot-v5',     # 50,51,52,53,54
            'ALE/Tutankham-v5', 	'ALE/UpNDown-v5', 			'ALE/Venture-v5', 		'ALE/VideoPinball-v5', 	'ALE/WizardOfWor-v5',   # 55,56,57,58,59
            'ALE/YarsRevenge-v5',                                                                                                       # 60
        ]

        self.DM_CONTROL_ENV_SEQS = list(suite.ALL_TASKS)
        

    @property
    def n_tasks(self):
        """int: the number of tasks."""
        return len(self._tasks)

    def sample(self, task_seq_idx):
        """Sample a list of environment updates.

        Note that this will always return environments in the same order, to
        make parallel sampling across workers efficient. If randomizing the
        environment order is required, shuffle the result of this method.

        Args:
            task_seq_idx: The index of tasks in the continual learning task sequence

        Raises:
            ValueError: If the number of requested tasks is not equal to the
                number of classes or the number of total tasks.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """

        updates = []

        # Avoid pickling the entire task sampler into every EnvUpdate
        inner_wrapper = self._inner_wrapper

        def metaworld_wrap(env, task):
            """Wrap an environment in a metaworld benchmark.

            Args:
                env (gym.Env): A metaworld / gym environment.
                task (metaworld.Task): A metaworld task.

            Returns:
                garage.Env: The wrapped environment.

            """
            env = GymEnv(env, max_episode_length=env.max_path_length)
            env = TaskNameWrapper(env, task_name=task.env_name)
            
            if inner_wrapper is not None:
                env = inner_wrapper(env, task)
            env = SuccessCounter(env)
            return env
        
        def gym_wrap(task_name):
            env = GymEnv(task_name)
            env = TaskNameWrapper(env, task_name=task_name)
            if inner_wrapper is not None:
                env = inner_wrapper(env)
            return env
        
        def atari_wrap(task_name, eval = False):
            env = gym.make(task_name)
            env = Noop(env, noop_max=30)
            env = MaxAndSkip(env, skip=4)
            env = EpisodicLife(env)
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireReset(env)
            env = Grayscale(env)
            env = Resize(env, 84, 84)
            if not eval:
                env = ClipReward(env)
            env = StackFrames(env, 4, axis=0)
            # env = GymEnv(env, max_episode_length=1000, is_image=True)
            # env = GymEnv(env, max_episode_length=10000, is_image=True)
            # env = GymEnv(env, max_episode_length=10800, is_image=True)
            env = GymEnv(env, max_episode_length=108000, is_image=True)

            return env


        if self._env_type == 'metaworld':
            env_seq = [self.MT50_ENV_SEQS[i] for i in task_seq_idx]
            for env_name in env_seq:
                env = self._classes[env_name]
                task = self._task_map[env_name][0]
                updates.append(SetTaskUpdate(env, task, metaworld_wrap))
        
        elif self._env_type == 'gym':
            env_seq = [self.GYM_ENV_SEQS[i] for i in task_seq_idx]
            for env_name in env_seq:
                env = gym_wrap(env_name)
                updates.append(env)

        elif self._env_type == 'dmlab':

            LEVELS = ['lt_chasm', 
                      'lt_hallway_slope',
                      'lt_horseshoe_color', 
                      'lt_space_bounce_hard',
                      'nav_maze_random_goal_01',
                      'nav_maze_random_goal_02', 
                      'nav_maze_random_goal_03',
                      'nav_maze_static_01', 
                      'nav_maze_static_02', 
                      'nav_maze_static_03',
                      'seekavoid_arena_01', 
                      'stairway_to_melon']
            
            colors = "RGB_INTERLEAVED"
            config = dict(
                width = str(84),
                height = str(84),
                fps = str(60),
            )

            level_seq = [LEVELS[i] for i in task_seq_idx]
            for level_name in level_seq:
                env = deepmind_lab.Lab(
                    level = level_name,
                    observations = [colors],
                    config = config
                )
                env = DeepmindLabEnv(
                    lab = env,
                    name = level_name,
                    colors = colors,
                    config = config
                )
                task_name = env._task_name
                env = GymEnv(env, max_episode_length=env.spec.max_episode_length)
                env = TaskNameWrapper(env, task_name = task_name)
                if inner_wrapper is not None:
                    env = inner_wrapper(env)
                updates.append(env)
        
        elif self._env_type == 'atari':
            env_seq = [self.ATARI_ENV_SEQS[i] for i in task_seq_idx]
            eval_updates = []
            for env_name in env_seq:
                env = atari_wrap(env_name)
                updates.append(env)
                env = atari_wrap(env_name, eval=True)
                eval_updates.append(env)

        elif self._env_type == "dm_control":
            task_seq = [self.DM_CONTROL_ENV_SEQS[i] for i in task_seq_idx]
            for domain, task in task_seq:
                env = DMControlEnv.from_suite(domain, task)
                task_name = dmc_task_name((domain, task))
                env = TaskNameWrapper(env, task_name = task_name)
                if inner_wrapper is not None:
                    env = inner_wrapper(env)
                updates.append(env)
            
        train_envs = [ContinualLearningEnv(updates, self._env_type, self._steps_per_env, self.seed, self._exploration_steps)]
        if self._env_type == 'atari':
            return train_envs, eval_updates
        
        return train_envs, updates
