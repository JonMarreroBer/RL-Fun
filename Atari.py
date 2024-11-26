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
agent = DQN("CnnPolicy", env, verbose=1)
agent.learn(total_timesteps=10_000)

# Create a folder to save videos
video_folder = "videos/"
os.makedirs(video_folder, exist_ok=True)

# Wrap environment for video recording
recording_env = VecVideoRecorder(env, video_folder,
                                 record_video_trigger=lambda x: x % 5000 == 0,
                                 video_length=200)  # Record every 5000 steps
obs = recording_env.reset()

# Record video
for _ in range(1000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, info = recording_env.step(action)
    if done.all():
        obs = recording_env.reset()
recording_env.close() 

print(f"Video saved in {video_folder}")