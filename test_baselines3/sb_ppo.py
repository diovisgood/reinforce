from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from init import *


def run(env_name: str,
        hidden_size=256,
        n_envs=4,
        total_timesteps=1000000,
        **kwargs
        ):
    # Prepare environment
    # Create the vectorized environment
    env = make_vec_env(env_name, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv)
    
    log.info(f'Env: {env}')

    # Initialize optimizer
    autosave_prefix = f'sb_ppo_{env_name.rsplit("-", 1)[0].lower()}'

    trainer = PPO('MlpPolicy', env, verbose=2)
    trainer.learn(total_timesteps=total_timesteps)

    obs = env.reset()
    for _ in range(1000):
        action, _states = trainer.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    # trainer = PPO(
    #     MlpPolicy,
    #     env,
    #     policy_kwargs=dict(activation_fn=th.nn.ELU, net_arch=[hidden_size, hidden_size]),
    #     verbose=2,
    #     **kwargs
    # )
    #
    # eval_env = gym.make(env_name)
    # eval_env = Monitor(eval_env)
    # eval_freq = max(1000, total_timesteps // 20)
    # eval_log_path = os.path.join(WORK_DIR, autosave_prefix)
    # eval_callback = EvalCallback(
    #     eval_env,
    #     log_path=eval_log_path,
    #     eval_freq=eval_freq,
    #     deterministic=True,
    #     render=True,
    #     n_eval_episodes=5
    # )
    #
    # trainer.learn(total_timesteps=total_timesteps, log_interval=500, callback=eval_callback)
    trainer.save(os.path.join(WORK_DIR, autosave_prefix))


def test(env_name: str):
    # Prepare environment
    env = get_env(env_name)
    log.info(f'Env: {env}')
    
    autosave_prefix = f'sb_ppo_{env_name.rsplit("-", 1)[0].lower()}.zip'
    model = PPO.load(os.path.join(WORK_DIR, autosave_prefix))
    
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    run(
        env_name='LunarLander-v2',
        hidden_size=256,
        n_envs=4,
        total_timesteps=200000,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.98,
        gamma=0.999,
        n_epochs=4,
        ent_coef=0.01,
    )
    
    test(env_name='LunarLander-v2')
