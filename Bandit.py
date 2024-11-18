"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv 

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    @classmethod
    def plot1(cls, rewards, num_trials, optimal_bandit_reward):
        """
        Plots the cummulative average reward convergence of bandit algorithms.

        :param rewards: A list of rewards obtained in each trial.
        :param num_trials: The total number of trials.
        :param optimal_bandit_reward: The reward of the optimal bandit.
        """
        cumulative_rewards = np.cumsum(rewards)
        average_reward = cumulative_rewards / (np.arange(num_trials) + 1)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(average_reward, label="Average Reward")
        ax[0].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[0].legend()
        ax[0].set_title("(Linear Scale)")
        ax[0].set_xlabel("Number of Trials")
        ax[0].set_ylabel("Average Reward")

        ax[1].plot(average_reward, label="Average Reward")
        ax[1].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[1].legend()
        ax[1].set_title("(Log Scale)")
        ax[1].set_xlabel("Number of Trials")
        ax[1].set_ylabel("Cummulative Reward")
        ax[1].set_yscale("log")
        
        fig.suptitle(f'Average Reward Convergence')

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot2(cls, rewards_eg, rewards_ts, num_trials, optimal_bandit_reward_eg, optimal_bandit_reward_ts):
        """
        Compares Epsilon-Greedy and Thompson Sampling cumulative rewards and regrets.

        :param rewards_eg: List of rewards obtained by Epsilon-Greedy in each trial.
        :param rewards_ts: List of rewards obtained by Thompson Sampling in each trial.
        :param num_trials: Total number of trials.
        :param optimal_bandit_reward_eg: The reward of the optimal bandit for Epsilon-Greedy.
        :param optimal_bandit_reward_ts: The reward of the optimal bandit for Thompson Sampling.
        """
        # Calculate cumulative rewards
        cumulative_rewards_eg = np.cumsum(rewards_eg)
        cumulative_rewards_ts = np.cumsum(rewards_ts)

        # Calculate cumulative regrets
        cumulative_regrets_eg = (
            optimal_bandit_reward_eg * np.arange(1, num_trials + 1) - cumulative_rewards_eg
        )
        cumulative_regrets_ts = (
            optimal_bandit_reward_ts * np.arange(1, num_trials + 1) - cumulative_rewards_ts
        )

        # Plot cumulative rewards comparison
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].plot(cumulative_rewards_eg, label="Epsilon-Greedy", color="blue")
        ax[0].plot(cumulative_rewards_ts, label="Thompson Sampling", color="green")
        ax[0].set_title("Cumulative Rewards (Linear Scale)")
        ax[0].set_xlabel("Number of Trials")
        ax[0].set_ylabel("Cumulative Reward")
        ax[0].legend()

        # Plot cumulative regrets comparison
        ax[1].plot(cumulative_regrets_eg, label="Epsilon-Greedy", color="blue")
        ax[1].plot(cumulative_regrets_ts, label="Thompson Sampling", color="green")
        ax[1].set_title("Cumulative Regrets (Linear Scale)")
        ax[1].set_xlabel("Number of Trials")
        ax[1].set_ylabel("Cumulative Regret")
        ax[1].legend()

        fig.suptitle("Comparison of Epsilon-Greedy and Thompson Sampling")
        plt.tight_layout()
        plt.show()


#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, m):
        """
        Initializes the EpsilonGreedy bandit.
        param m: the true mean reward for this arm (mean of the reward distribution).
        """
        self.m = m # the true mean reward for this arm
        self.m_estimate = 0. # the mean reward estimate, initially set to 0
        self.N = 0 # the number of times this arm has been pulled 
   
    def pull(self):
        """
        "Pull" the arm of the bandit.

        Returns:
        float: the observed reward, sampled from a normal distribution with mean `self.p` and unit variance.
        """
        return np.random.random() + self.m
    
    def update(self, x):
        """
        Update the bandit's estimated reward based on the observed reward.

        param x: the observed reward from the most recent pull.
        """
        self.N += 1
        self.m_estimate = (1 - 1.0/self.N)*self.m_estimate + 1.0/self.N*x
         
    def __repr__(self):
        """
        A string representation of the bandit arm that shows its true mean reward.

        Returns:
        str: a description of the arm's true mean reward.
        """
        return f"EpsilonGreedy Arm with true mean {self.p:.2f}"
    
    @classmethod
    def experiment(cls, bandit_return, num_trials, initial_epsilon = 0.1, min_epsilon = 0.0000001):
        """
        Run an experiment with the Epsilon-Greedy algorithm and decaying epsilon.

        param bandit_return: List of true mean rewards for each bandit.
        param initial_epsilon: Initial exploration rate (epsilon).
        param min_epsilon: Minimum exploration rate to ensure some exploration.
        param num_trials: Total number of trials in the experiment.

        return: a list of bandits and their corresponding rewards
        """
        bandits = [EpsilonGreedy(p) for p in bandit_return]

        # Initialize the rewards and counters
        rewards = []
        bandits = [EpsilonGreedy(m) for m in bandit_return]
        optimal_bandit = np.argmax([b.m for b in bandits])

        for i in range(1, num_trials + 1):
            epsilon = max(initial_epsilon / i, min_epsilon)

            if np.random.random() < epsilon:
                chosen_bandit = np.random.randint(len(bandits))
            else:
                chosen_bandit = np.argmax([b.m_estimate for b in bandits])

            reward = bandits[chosen_bandit].pull()
            rewards.append(reward)
            bandits[chosen_bandit].update(reward)


        return bandits, rewards

    @classmethod
    def report(cls, bandit_rewards, num_trials):
        """
        Generates a report for the Epsilon-Greedy algorithm.
        Saves the rewards for each trial in a CSV file in the format {Bandit, Reward, Algorithm}.
        Plots the rewards and regrets over the number of trials.
        Calculates and prints the total reward and total regret.

        :param bandit_rewards: A list of rewards for each bandit.
        :param num_trials: The number of trials to run the experiment.
        """
        # Run the experiment
        bandits, rewards = cls.experiment(bandit_return = bandit_rewards,  num_trials = num_trials)

        # Determine the optimal bandit reward
        optimal_bandit_reward = max(bandit_rewards)

        # Open the CSV file in append mode
        with open("results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            # Write header only if the file is empty
            if f.tell() == 0:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            
            # Write each trial's result to the CSV file
            for i, reward in enumerate(rewards):
                chosen_bandit = i % len(bandits)
                writer.writerow([chosen_bandit, reward, "Epsilon Greedy"])

        # Log the estimates for each bandit
        for i, b in enumerate(bandits):
            logger.info(
                f"Bandit {i} - Estimated Mean: {b.m_estimate:.2f}, "
                f"True Mean: {b.m:.2f}, Times Pulled: {b.N}"
            )

        # Plot cumulative rewards and regrets
        Visualization.plot1(rewards, num_trials, optimal_bandit_reward)

        # Calculate and log total rewards and regrets
        total_reward = sum(rewards)
        total_regret = optimal_bandit_reward * num_trials - total_reward
        logger.info(f"Total Reward: {total_reward:.2f}")
        logger.info(f"Total Regret: {total_regret:.2f}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, m):
        """
        param m: the true mean for the bandit
        """
        self.m = m
        
        # Parameters for mu - prior is N(0,1)
        self.m_estimate = 0
        self.lambda_ = 1
        
        # Precision
        self.tau = 1 #we set it ourselves
        self.sum_x = 0
    
        # Number of times the bandit has been pulled
        self.N = 0

    def pull(self):
        """
        Pulling the bandit arm and returning a random number from a normal distribution.
        
        :return(float): a random number from a normal distribution with mean m.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.m
    
    def sample(self):
        """
        Pulling the bandit arm and returning a random number from a normal distribution (with an updated mean).
        
        :return(float): a random number from a normal distribution with the estimated mean.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m_estimate

    def update(self, x):
        """
        Updating the bandit's distribution parameters based on the observed reward.

        :param x: the observed reward
        """
        # increase the number of times the arm has been pulled
        self.N += 1
        
        # update the parameters of the normal distribution
        self.lambda_ += self.tau
        self.sum_x += x
        self.m_estimate = (self.tau * self.sum_x)/self.lambda_
        
    def __repr__(self):
        """Return a string representation of the bandit."""
        return f"An Arm with {self.m} Reward"

    @classmethod
    def experiment(cls, bandit_rewards, num_trials):
        """
        Run an experiment using the Thompson Sampling algorithm.

        :param bandit_rewards: A list of true means for each bandit arm.
        :param num_trials: The number of trials to run the experiment.
        :return: The bandits and the rewards obtained during the experiment.
        """

        bandits = [ThompsonSampling(m) for m in bandit_rewards]
        
        sample_points = [5, 10, 100, 1000, 5000]
        
        rewards = []

        for i in range(num_trials):
            # Choose the bandit with the highest sampled value
            j = np.argmax([b.sample() for b in bandits])
                
            # Pull the chosen bandit
            x = bandits[j].pull()
            
            # Add the reward to the list of rewards
            rewards.append(x)
            
            # Update the chosen bandit
            bandits[j].update(x)
        
        return bandits, rewards
    
    @classmethod
    def report(cls, bandit_rewards, num_trials):
        """
        Generate a report for the Thompson Sampling experiment.

        Parameters:
            bandit_probabilities (list): List of true means for each bandit.
            num_trials (int): Number of trials to run.
        """
        # run the experiment 
        bandits, rewards = ThompsonSampling.experiment(bandit_rewards = bandit_rewards, num_trials = num_trials)
        with open("results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            # Write header only if the file is empty
            if f.tell() == 0:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            
            # Append trial results to the file
            for i, reward in enumerate(rewards):
                chosen_bandit = i % len(bandits)  # Bandit responsible for this trial's reward
                writer.writerow([chosen_bandit, reward, "Thompson Sampling"])
        logger.info("Appended the results to results.csv.")

        optimal_bandit_reward = max(bandit_rewards)

        Visualization.plot1(rewards, num_trials, optimal_bandit_reward)

        total_reward = sum(rewards)
        total_regret = optimal_bandit_reward * num_trials - total_reward
        logger.info(f"Total Reward: {total_reward:.2f}; Average Reward: {total_reward/num_trials:.2f}")
        logger.info(f"Total Regret: {total_regret:.2f}; Average Regret: {total_regret/num_trials:.2f}")




def comparison(bandit_rewards, num_trials):
    """
    Compare the performance of Epsilon-Greedy and Thompson Sampling algorithms visually.

    :param bandit_rewards: List of true means for each bandit arm.
    :param num_trials: Total number of trials to run the experiments.
    """
    logger.info("Running Epsilon-Greedy experiment...")
    _, rewards_eg = EpsilonGreedy.experiment(bandit_return=bandit_rewards, num_trials=num_trials)

    logger.info("Running Thompson Sampling experiment...")
    _, rewards_ts = ThompsonSampling.experiment(bandit_rewards=bandit_rewards, num_trials=num_trials)

    # Determine the optimal rewards for both algorithms
    optimal_bandit_reward = max(bandit_rewards)

    # Use Visualization class to plot the comparison
    Visualization.plot2(
        rewards_eg,
        rewards_ts,
        num_trials,
        optimal_bandit_reward,
        optimal_bandit_reward,
    )

    logger.info("Comparison between Epsilon-Greedy and Thompson Sampling completed.")


if __name__ == "__main__":
    bandit_rewards = [1.0, 2.0, 3.0, 4.0]
    num_trials = 20000

    EpsilonGreedy.report(bandit_rewards, num_trials)
    ThompsonSampling.report(bandit_rewards, num_trials)
    comparison(bandit_rewards, num_trials)
