import gymnasium as gym
from stable_baselines3 import   A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import numpy as np
from time import time
from stable_baselines3.common.monitor import Monitor
import highway_env  # noqa: F401

experiment_name = "F23-A2C-CNN"
num_episodes = 50000
actor_learning_rate = 1.5e-4 #5e-4
critic_learning_rate = 7.75e-2
gamma = 0.75 # discount factor

def train_env():
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.configure(
        {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
        }
    )
    #env.config["reward_speed_range"] = [10, 20]
    #env.config["collision_reward"] = -2
    env.reset()
    return Monitor(env)


def test_env():
    env = train_env()
    env.configure({"duration": 1000})
    env.reset()
    return env


if True:

    # Train
    model = A2C(
        "CnnPolicy",
        DummyVecEnv([train_env]),
        learning_rate=actor_learning_rate, # funcion de valor, #actor_learning_rate
        gamma=gamma,
        verbose = 1,
        tensorboard_log=experiment_name + "/logs",
        vf_coef = 1,
        max_grad_norm = 5,
        normalize_advantage = True,
        rms_prop_eps = True
    )
    print(model.policy.optimizer.param_groups[0]['lr'])
    model.policy.optimizer.param_groups[0]['lr']=critic_learning_rate
    print(model.policy)
    model.learn(total_timesteps=int(num_episodes), progress_bar=True)
    model.save(experiment_name + "/model")
    

    model = A2C.load(experiment_name + "/model")

    env = DummyVecEnv([test_env])

    obs = env.reset()

    metrics = {
    "time_alive":[],
    "mean_rewards":[],
    "average_speed":[]
    }

    try:
        for i in range(10):
            # We use mean of time alive of 10 experiment as a metric
            time_start = time()
            done = truncated = False
            obs = env.reset()
            rewards = []
            speed = []
            while not (done):
                action, _states = model.predict(obs, deterministic=True)
                print(action)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                speed.append(info[0]['speed']) # speed of the car
                env.render()
            metrics["mean_rewards"].append(np.mean(rewards))
            metrics["time_alive"].append(time() - time_start)
            metrics["average_speed"].append(np.mean(speed))
        print(f"mean time alive = {np.mean(metrics['time_alive'])}")
        print(f"mean rewards = {np.mean(metrics['mean_rewards'])}")
        print(f"average speed = {np.mean(metrics['average_speed'])}")

    except:
        metrics["mean_rewards"].append(np.mean(rewards))
        metrics["time_alive"].append(time() - time_start)
        metrics["average_speed"].append(np.mean(speed))
        
        print(f"mean time alive = {np.mean(metrics['time_alive'])}")
        print(f"mean rewards = {np.mean(metrics['mean_rewards'])}")
        print(f"average speed = {np.mean(metrics['average_speed'])}")