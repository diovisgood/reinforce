"""
    Implementation of Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE)
    This module is designed to train a model for standard Gym environments.
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - July 2021
    License: MIT
"""
from init import *
from optim.models import Models
from optim.dqn import DQN
from optim.scheduler import Scheduler


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
    if kwargs.get('dueling_mode', False):
        # For duelling mode: adjust output_shape so that model will output a tuple of: (advantages, value)
        output_shape = (space_shape(env.action_space), 1)
    else:
        # For vanilla DQN a model outputs qvalues directly
        output_shape = space_shape(env.action_space)
    model_class = Models[model_name]
    model = model_class(input_shape=input_shape, output_shape=output_shape, hidden_size=hidden_size, norm=False)
    log.info(f'Model: {model}')

    # Initialize optimizer
    autosave_prefix = f'dqn_{model_name.lower()}_{env_name.rsplit("-", 1)[0].lower()}'
    trainer = DQN(
        env=env,
        model=model,
        step_delay=0.015,
        autosave_dir=WORK_DIR,
        autosave_prefix=autosave_prefix,
        autosave_interval=timedelta(minutes=1),
        log=log,
        **kwargs
    )
    log.info(f'Trainer: {trainer}')
    
    trainer.fit(total_timesteps=total_timesteps, render=True)
    trainer.autosave(force=True)


def test(env_name: str, model_name: str, hidden_size=256, **kwargs):
    # Prepare environment
    env = get_env(env_name)
    log.info(f'Env: {env}')
    
    # Initialize model
    input_shape = space_shape(env.observation_space)
    if kwargs.get('dueling_mode', False):
        # For duelling mode: adjust output_shape so that model will output a tuple of: (advantages, value)
        output_shape = (space_shape(env.action_space), 1)
    else:
        # For vanilla DQN a model outputs qvalues directly
        output_shape = space_shape(env.action_space)
    model_class = Models[model_name]
    model = model_class(input_shape=input_shape, output_shape=output_shape, hidden_size=hidden_size, norm=False)
    log.info(f'Model: {model}')
    
    # Load model
    autosave_prefix = f'dqn_{model_name.lower()}_{env_name.rsplit("-", 1)[0].lower()}'
    model_file_name = os.path.join(WORK_DIR, autosave_prefix + '.model.pt')
    model.load_state_dict(th.load(model_file_name), strict=True)
    
    state = env.reset()
    while True:
        action = model(state)
        action = action.argmax(dim=-1).squeeze().item()
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            state = env.reset()


if __name__ == '__main__':
    # run(env_name='CartPole-v0', model_name='fc2')
    # run(env_name='Acrobot-v1', model_name='fc2', entropy_factor=1.0)
    # run(env_name='BreakoutDeterministic-v4', model_name='convfc1', hidden_size=128)
    run(
        env_name='LunarLander-v2',
        model_name='fc2',
        double_mode=True,
        dueling_mode=False,
        learning_rate=Scheduler((5e-4, 1e-7), 's'),
        gamma=0.99,
        batch_size=128,
        total_timesteps=150000,
        anneal_period=int(100000 * 0.12),
        prioritized_replay=True,
        replay_buffer_capacity=50000,
        samples_to_start=10000,
        samples_per_iteration=4,
        updates_per_iteration=4,
        target_update_period=250
    )
    test(env_name='LunarLander-v2', model_name='fc2')
