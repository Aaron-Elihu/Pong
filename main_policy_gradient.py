import pygame
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2  # opencv-python
from settings import width, height
from table import Table


# Environment using PyGame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ping Pong")

# Hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint
render = False


class Pong:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        # Initialize your game object (Table) that handles game logic
        self.table = Table(self.screen)
        # Initialize agent variables
        self.model = self.init_model()  # e.g., {'W1': ..., 'W2': ...}
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rmsp_cache = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.learning_rate = 1e-4
        self.decay_rate = 0.99
        self.gamma = 0.99  # discount factor for reward
        self.batch_size = 10
        # For recording training stats
        self.reward_history = []
        self.running_mean_rewards = []
        self.window_size = 10  # for running mean calculation

    @staticmethod
    def init_model():
        # Example model initialization (adjust dimensions as needed)
        D = 6400  # e.g., input dimension (80x80 image flattened)
        H = 200   # hidden layer size
        model = {
            'W1': np.random.randn(H, D) / np.sqrt(D),
            'W2': np.random.randn(H) / np.sqrt(H)
        }
        return model

    def draw(self):
        self.screen.fill("black")
        self.table.draw(self.screen)
        self.table.human_player.draw(self.screen)
        self.table.ai_player.draw(self.screen)
        pygame.display.flip()

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def preprocess(img):
        img = pygame.surfarray.array3d(img)  # Convert Pygame surface to NumPy
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (80, 80))  # Resize to 80x80
        img = img / 255.0  # Normalize pixel values
        return img.astype(np.float32).ravel()  # Flatten into (6400,)

    @staticmethod
    def discount_rewards(r):
        # take 1D float array of rewards and compute discounted reward
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU non-linearity
        log_p = np.dot(self.model['W2'], h)
        p = self.sigmoid(log_p)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, ep_x, ep_hidden, ep_dlog_p):  # episode_hidden, episode_differences log p (gradient)
        # backward pass. (ep_hidden is array of intermediate hidden states)
        dW2 = np.dot(ep_hidden.T, ep_dlog_p).ravel()  # one-dimension vector of dot product
        dh = np.outer(ep_dlog_p, self.model['W2'])  # gradient loss wrt. hidden layer output
        dh[ep_hidden <= 0] = 0  # backpropagation RELU
        dW1 = np.dot(dh.T, ep_x)  # episode input data
        return {'W1': dW1, 'W2': dW2}  # dictionary of gradients for both layers

    @staticmethod
    def moving_average(values, window_size=10):
        return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

    def run_episode(self):
        self.table.reset()
        observation = self.table.get_observation()

        previous_x = None
        xs, hs, dlog_ps, drs = [], [], [], []  # store episode history
        episode_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    continue
            self.table.player_move()
            self.table.update()

            current_x = Pong.preprocess(observation)
            x = current_x - previous_x if previous_x is not None else np.zeros_like(current_x)
            previous_x = current_x

            # Get action from policy
            a_prob, h = self.policy_forward(x)
            # Sample action (assuming two actions: 2 and 3)
            action = 2 if np.random.uniform() < a_prob else 3

            # Record intermediates for learning
            xs.append(x)
            hs.append(h)
            y = 1 if action == 2 else 0
            dlog_ps.append(y - a_prob)

            # Step the game forward
            reward, done = self.table.step(action)
            episode_reward += reward
            drs.append(reward)
            if done:
                done = False

            # Update game visuals
            self.screen.fill("black")
            self.table.update()
            self.draw()
            self.clock.tick(30)
            observation = self.table.get_observation()

        # End of episode: perform learning updates
        ep_x = np.vstack(xs)
        ep_h = np.vstack(hs)
        ep_dlog_p = np.vstack(dlog_ps)
        ep_r = np.vstack(drs)

        # Discount rewards
        discounted_ep_r = self.discount_rewards(ep_r)
        discounted_ep_r -= np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)
        ep_dlog_p *= discounted_ep_r

        # Backpropagation to compute gradients
        grad = self.policy_backward(ep_x, ep_h, ep_dlog_p)
        for k in self.model:
            self.grad_buffer[k] += grad[k]

        # Update model every batch_size episodes
        if (len(self.reward_history) + 1) % self.batch_size == 0:
            for k, v in self.model.items():
                g = self.grad_buffer[k]
                self.rmsp_cache[k] = self.decay_rate * self.rmsp_cache[k] + (1 - self.decay_rate) * g**2
                self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsp_cache[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v)

        return episode_reward

    def train(self, num_episodes=1000):
        plt.ion()  # enable interactive plotting
        fig, ax = plt.subplots(figsize=(20, 8))
        running_reward = None

        for ep in range(num_episodes):
            episode_reward = self.run_episode()
            # Update running reward (exponential moving average)
            running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01

            self.reward_history.append(episode_reward)
            if len(self.reward_history) >= self.window_size:
                current_running_mean = np.mean(self.reward_history[-self.window_size:])
            else:
                current_running_mean = np.mean(self.reward_history)
            self.running_mean_rewards.append(current_running_mean)

            print(f"Episode {ep+1}: Reward {episode_reward: .6f}, Running Mean {current_running_mean: .6f}")

            # Update dynamic plot
            ax.clear()
            ax.plot(self.reward_history, label="Episode Reward", alpha=0.5)
            ax.plot(self.running_mean_rewards, label="Running Mean Reward", color="red", linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.set_title("Learning Curve")
            ax.legend()
            plt.pause(0.001)

            # Save the model every 100 episodes
            if (ep + 1) % 100 == 0:
                pickle.dump(self.model, open('model.pkl', 'wb'))

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    pong_game = Pong(screen)
    pong_game.train(num_episodes=30000)
