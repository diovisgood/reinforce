from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


from init import *


def run(env_name: str,
        hidden_size=256,
        total_timesteps=200000,
        **kwargs
        ):
    # Prepare environment
    env = get_env(env_name)
    env = Monitor(env)
    log.info(f'Env: {env}')

    # Initialize optimizer
    autosave_prefix = f'sb_a2c_{env_name.rsplit("-", 1)[0].lower()}'

    trainer = A2C(
        MlpPolicy,
        env,
        policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[hidden_size, hidden_size]),
        verbose=2,
        **kwargs
    )

    eval_env = gym.make(env_name)
    eval_env = Monitor(eval_env)
    eval_freq = max(1000, total_timesteps // 20)
    eval_log_path = os.path.join(WORK_DIR, autosave_prefix)
    eval_callback = EvalCallback(
        eval_env,
        log_path=eval_log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=True,
        n_eval_episodes=5
    )

    trainer.learn(total_timesteps=total_timesteps, log_interval=500, callback=eval_callback)
    trainer.save(os.path.join(WORK_DIR, autosave_prefix))


def test(env_name: str):
    # Prepare environment
    env = get_env(env_name)
    log.info(f'Env: {env}')
    
    autosave_prefix = f'sb_a2c_{env_name.rsplit("-", 1)[0].lower()}.zip'
    model = A2C.load(os.path.join(WORK_DIR, autosave_prefix))
    
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
        total_timesteps=200000,
        gamma=0.99,
        n_steps=5,
    )
    
    test(env_name='LunarLander-v2')
