"""
    Implementation of Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE)
    This module is designed to train a model for standard Gym environments.
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - July 2021
    License: MIT
"""
from init import *
from optim.models import Models
from optim.a3c import A3C


def run(env_name: str,
        model_name: str,
        hidden_size=256,
        total_timesteps=200000,
        **kwargs):
    # Prepare environment
    env = get_env(env_name)
    log.info(f'Env: {env}')

    # Initialize model
    input_shape = space_shape(env.observation_space)
    # Adjust output_shape so that model will output a tuple of: (advantages, value)
    output_shape = (space_shape(env.action_space), 1)
    model_class = Models[model_name]
    model = model_class(input_shape=input_shape, output_shape=output_shape, hidden_size=hidden_size, norm=False)
    log.info(f'Model: {model}')

    # Initialize optimizer
    autosave_prefix = f'a3c_{model_name.lower()}_{env_name.rsplit("-", 1)[0].lower()}'
    trainer = A3C(
        env=env,
        model=model,
        step_delay=0.012,
        autosave_dir=WORK_DIR,
        autosave_prefix=autosave_prefix,
        autosave_interval=timedelta(minutes=1),
        log=log,
        **kwargs
    )
    log.info(f'Trainer: {trainer}')

    # score = trainer.test(render=True, max_steps=1000)
    # print(f'score={score}')
    trainer.fit(total_timesteps=total_timesteps, render=True)
    trainer.autosave(force=True)


if __name__ == '__main__':
    # run(env_name='CartPole-v0', model_name='fc2')
    # run(env_name='Acrobot-v1', model_name='fc2', entropy_factor=1.0)
    # run(env_name='BreakoutDeterministic-v4', model_name='convfc1', hidden_size=128)
    # run(
    #     env_name='LunarLander-v2',
    #     model_name='fc2',
    #     hidden_size=256,
    #     learning_rate=6e-4,
    #     max_samples=800000,
    #     n_workers=5,
    #     steps_per_update=5,
    #     gamma=0.995,
    #     entropy_factor=0.1,
    # )

    run(
        env_name='Acrobot-v1',
        model_name='fc2',
        hidden_size=256,
        total_timesteps=510000,
        n_workers=5,
        steps_per_update=5,
    )
