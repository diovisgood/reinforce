"""
    Demonstration code to run Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - August 2021
    License: MIT
"""
from init import *
from optim.ppo import PPO
from optim.models import Models, BaseModel, MLP
from optim.agent import AgentActorCritic
from optim.scheduler import Scheduler


def run(continue_last: bool,
        env_name: str,
        actor_class: Type[BaseModel],
        actor_kwargs: Optional[Mapping],
        critic_class: Optional[Type[BaseModel]] = None,
        critic_kwargs: Optional[Mapping] = None,
        total_timesteps=200000,
        **kwargs):
    # Prepare environment
    env = get_env(env_name)
    log.info(f'Env: {env}')

    # Initialize agent
    agent = AgentActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_class=actor_class,
        actor_kwargs=actor_kwargs,
        critic_class=critic_class,
        critic_kwargs=critic_kwargs,
        normalize_input=False
    )
    log.info(f'Agent: {agent}')

    # Get either new or last directory for saving checkpoints
    autosave_dir = None
    if continue_last:
        autosave_dir = get_last_output_dir('ppo', env_name)
    if autosave_dir is None:
        autosave_dir = get_new_output_dir('ppo', env_name)

    # Initialize trainer
    trainer = PPO(
        env=env,
        agent=agent,
        step_delay=0.002,
        autosave_dir=autosave_dir,
        autosave_interval=timedelta(minutes=1),
        log=log,
        **kwargs
    )
    log.info(f'Trainer: {trainer}')

    # score = trainer.test(render=True, max_steps=1000)
    # print(f'score={score}')
    trainer.train(total_timesteps=total_timesteps, render=True)
    trainer.autosave(force=True)


if __name__ == '__main__':
    # run(env_name='CartPole-v0', model_name='fc2')
    # run(env_name='Acrobot-v1', model_name='fc2', entropy_factor=1.0)
    # run(env_name='BreakoutDeterministic-v4', model_name='convfc1', hidden_size=128)
    # run(
    #     env_name='Acrobot-v1',
    #     model_name='fc2',
    #     hidden_size=256,
    #     max_samples=500000,
    #     n_workers=5,
    #     steps_per_update=5,
    # )
    
    # run(
    #     env_name='CartPole-v1',
    #     agent_name='ac_fc2',
    #     hidden_size=256,
    #     total_timesteps=100000,
    #     n_workers=4,
    #     steps_per_iteration=200,
    #     update_timeout=timedelta(hours=5),
    # )
    run(
        continue_last=False,
        env_name='LunarLander-v2',
        actor_class=Models['lstm1'],
        actor_kwargs=dict(layers=128, init='orthogonal', activation_fn='tanh'),
        critic_class=Models['lstm1'],
        critic_kwargs=dict(layers=128, init='orthogonal', activation_fn='tanh'),
        total_timesteps=500000,
        learning_rate=Scheduler((3e-4, 1e-5), 's'),
        n_workers=2,
        steps_per_iteration=-3,
        epochs_per_iteration=50,
        recurrent=True,
        value_loss='mse',
        limit_kl_divergence=0.06,
        update_timeout=timedelta(hours=5),
    )
    # TODO: Try SELU and GELU activation_fn
    