"""
    Implementation of Covariance Matrix Adaptation Evolution Strategy (CMA-ES) in Pytorch
    This module is designed to train a model for standard Gym environments.
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - June 2021
    License: MIT
"""
import copy
import torch.nn as nn
import multiprocessing as mp

from init import *

from optim.cmaes import CMAES
from optim.models import Models


def scoring_function(env: gym.Env,
                     model: nn.Module,
                     population: Sequence[torch.Tensor],
                     n_iter: int,
                     episodes_per_iteration: int = 10,
                     cpu_limit: int = 2
                     ) -> Sequence[float]:
    """Evaluate a set of different possible weights for a model"""
    # Initialize a list of models with weights from specified population
    models = []
    for x in population:
        entity_model = copy.deepcopy(model)
        torch.nn.utils.vector_to_parameters(x, entity_model.parameters())
        models.append(entity_model)
        
    # Construct batch
    args = []
    for model in models:
        for i in range(episodes_per_iteration):
            seed = n_iter * episodes_per_iteration + i
            args.append((env, model, seed, False))
    
    # Run parallel or sequential
    if cpu_limit > 1:
        with mp.Pool(cpu_limit) as pool:
            rewards = pool.starmap(run_episode, args)
    else:
        rewards = [run_episode(*a) for a in args]
        
    # Compute mean reward for each entity over episodes
    scores = np.array(rewards).reshape(len(models), episodes_per_iteration).mean(axis=1, dtype=np.float).tolist()
    
    # Sometimes show screen
    if n_iter % 2 == 0:
        run_episode(env, model, render=True)

    return scores


def run_episode(env: gym.Env, model: nn.Module, seed=None, render=False) -> float:
    """Run new episode using provided seed (if any)"""
    env.seed(seed)
    state = env.reset()
    
    rewards = []
    while True:
        if render:
            env.render()
        action = model(state)
        action = convert_action(action, env.action_space)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break

    env.close()
    return float(np.sum(rewards))

    
def run(env_name: str, model_name: str, hidden_size=16, max_iter=200):
    # Prepare environment
    env = get_env(env_name)
    log.info(f'Env: {env}')
    
    # Initialize model
    input_shape = space_shape(env.observation_space)
    output_shape = space_shape(env.action_space)
    model_class = Models[model_name]
    model = model_class(input_shape=input_shape, output_shape=output_shape, hidden_size=hidden_size, norm=False)
    log.info(f'Model: {model}')

    # Initialize optimizer
    autosave_prefix = f'cmaes_{model_name.lower()}_{env_name.split("-", 1)[0].lower()}'
    optimizer = CMAES(
        initial_point=torch.nn.utils.parameters_to_vector(model.parameters()),
        scoring_function=lambda population, optimizer:
            scoring_function(env, model, population, n_iter=optimizer.n_iter, cpu_limit=N_WORKERS),
        autosave_dir=WORK_DIR,
        autosave_prefix=autosave_prefix,
        autosave_interval=timedelta(minutes=1),
        log=log
    )
    log.info(f'Optimizer: {optimizer}')
    
    # Run the optimization
    optimizer.fit(max_iter=max_iter)
    optimizer.autosave(force=True)


if __name__ == '__main__':
    # run(env_name='CartPole-v0', model_name='fc1')
    
    # Breakout fails to run due to large model size caused by large input size
    # run(env_name='Breakout-v0', model_name='fc1')
    
    # run(env_name='MountainCar-v0', model_name='fc1', max_iter=500)
    
    # run(env_name='MountainCarContinuous-v0', model_name='fc1')

    # run(env_name='Acrobot-v1', model_name='fc1', max_iter=100)

    # run(env_name='LunarLander-v2', model_name='gru1', max_iter=100)

    run(env_name='Breakout-v0', model_name='convgru1', hidden_size=32)
