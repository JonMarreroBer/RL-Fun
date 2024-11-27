from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
import gymnasium
import ale_py
import os

# Environment creation
env = make_atari_env("SpaceInvadersNoFrameskip-v4",n_envs=4,seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# Create the agent and train it
agent = DQN("CnnPolicy", env, verbose=1,buffer_size=100_000, learning_starts=100_000, train_freq=4, 
            target_update_interval=1_000, exploration_fraction=0.1, exploration_final_eps=0.01,
            batch_size=32, learning_rate=1e-4, gradient_steps=1, optimize_memory_usage=False)
agent.learn(total_timesteps=10_000)

# Create a folder to save videos
video_folder = "videos/"
os.makedirs(video_folder, exist_ok=True)

# Wrap environment for video recording
recording_env = VecVideoRecorder(env, video_folder,
                                 record_video_trigger=lambda x: True,
                                 video_length=1000) # Record every 5000 steps
obs = recording_env.reset()

# Record video
for _ in range(1000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, info = recording_env.step(action)
    if done.all():
        obs = recording_env.reset()
recording_env.close() 

print(f"Video saved in {video_folder}")