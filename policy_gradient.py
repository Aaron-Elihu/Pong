# Implement a Policy Gradient Agent designed to learn Pong in OpenAI Gym
import numpy as np
import pickle
import gymnasium as gym
import ale_py  # for ALE environment
import matplotlib.pyplot as plt


# Environment using Atari
env = gym.make("ALE/Pong-v5")
observation, info = env.reset()

# Hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint
render = False

# Model initialization
D = 6400  # input dimensionality: 80x80 grid flattened
if resume:
    model = pickle.load(open('model.pkl', 'rb'))  # pickle file
else:
    model = {'W1': np.random.randn(H, D) / np.sqrt(D), 'W2': np.random.randn(H) / np.sqrt(H)}

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsp_cache = {k: np.zeros_like(v) for k, v in model.items()}  # root-mean-square propagation memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(i):
    # preprocess 210*160x3 (max resolution) uint8 frame into 64000 (80x80) 1D float vector
    # crop, downsample, remove unwanted background colours, binarize and flatten image
    i = i[35:195]  # crop to remove unnecessary pixels: scoreboard, paddles and board.
    i = i[::2, ::2, 0]  # downsample to 80x80 for 2D array
    i[i == 144] = 0  # erase background (background type 1)
    i[i == 109] = 0  # erase background (background type 2)
    i[i != 0] = 1  # everything else (paddles, ball), set to 1
    return i.astype(np.float64).ravel()


def discount_rewards(r):
    # take 1D float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU non-linearity
    log_p = np.dot(model['W2'], h)
    p = sigmoid(log_p)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(ep_hidden, ep_dlog_p):  # episode_hidden, episode_differences log p (gradient)
    # backward pass. (ep_hidden is array of intermediate hidden states)
    dW2 = np.dot(ep_hidden.T, ep_dlog_p).ravel()  # one-dimension vector of dot product
    dh = np.outer(ep_dlog_p, model['W2'])  # gradient loss wrt. hidden layer output
    dh[ep_hidden <= 0] = 0  # backpropagation RELU
    dW1 = np.dot(dh.T, ep_x)  # episode input data
    return {'W1': dW1, 'W2': dW2}  # dictionary of gradients for both layers


# Define the smoothing function
def moving_average(values, window_size=10):
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')


# Main training loop for Policy Gradient Agent
previous_x = None  # used to compute the difference of consecutive frames (highlights motion)
xs, hs, dlog_ps, drs = [], [], [], []  # preprocessed frame inputs, hidden states, gradients of log probabilities, rewards received at each timestep
running_reward = None
reward_sum = 0
episode_number = 0

reward_history = []  # stores episode rewards
running_mean_rewards = []  # stores smoothed rewards
window_size = 10          # window for running mean calculation

# Set up dynamic plot
plt.ion()  # interactive mode on
fig, ax = plt.subplots(figsize=(20, 8))

max_episodes = 30000  # training stops after 30000 episodes

while episode_number < max_episodes:
    if render:  # render if needed
        env.render()

    # Preprocess the observation
    current_x = preprocess(observation)
    # Compute difference frame
    x = current_x - previous_x if previous_x is not None else np.zeros(D)
    previous_x = current_x

    # Forward pass through policy network
    a_prob, h = policy_forward(x)
    action = 2 if np.random.uniform() < a_prob else 3

    # Record intermediates
    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0
    dlog_ps.append(y - a_prob)

    # Step environment
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    reward_sum += reward
    drs.append(reward)

    if done:
        episode_number += 1

        # Stack episode data
        ep_x = np.vstack(xs)
        ep_h = np.vstack(hs)
        ep_dlog_p = np.vstack(dlog_ps)
        ep_r = np.vstack(drs)
        xs, hs, dlog_ps, drs = [], [], [], []  # reset episode storage

        # Compute discounted rewards
        discounted_ep_r = discount_rewards(ep_r)
        discounted_ep_r -= np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)
        ep_dlog_p *= discounted_ep_r

        # Backpropagation
        grad = policy_backward(ep_h, ep_dlog_p)
        for k in model:
            grad_buffer[k] += grad[k]

        # Update weights using RMSProp every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsp_cache[k] = decay_rate * rmsp_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsp_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        # Update running reward (exponential moving average)
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f"resetting environment episode reward total was {reward_sum: .6f}. running mean: {running_reward: .6f}")

        # Save model periodically
        if episode_number % 100 == 0:
            pickle.dump(model, open('model.pkl', 'wb'))

        # Store rewards for plotting
        reward_history.append(reward_sum)
        if len(reward_history) >= window_size:
            running_mean_rewards.append(np.mean(reward_history[-window_size:]))
        else:
            running_mean_rewards.append(np.mean(reward_history))

        # Update dynamic plot
        ax.clear()
        ax.plot(reward_history, label="Episode Reward", alpha=0.5)
        ax.plot(running_mean_rewards, label="Running Mean Reward", color="red", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Learning Curve")
        ax.legend()
        plt.pause(0.001)

        # Reset reward sum and previous frame
        reward_sum = 0
        observation, info = env.reset()
        previous_x = None

        # Optionally print the game finished message (if reward != 0)
        if reward != 0:
            print(f"ep {episode_number}: game finished, reward: {reward: .6f}" + ("" if reward == -1 else " !!!!!!!!"))

plt.ioff()
plt.show()
