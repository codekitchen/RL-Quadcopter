import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.2, steps=1):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.steps = steps
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu
        self.steps_left = 1

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.steps_left = self.steps
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
            self.state = x + dx
        return self.state
