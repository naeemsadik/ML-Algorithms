import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class RecurrentNeuralNetwork:
    def __init__(self, hidden_size=80, spectral_radius=0.9, regularization=1e-5, random_state=42):
        self.hidden_size = hidden_size
        self.spectral_radius = spectral_radius
        self.regularization = regularization
        self.random_state = random_state

    def _initialize(self):
        rng = np.random.default_rng(self.random_state)
        self.input_weights_ = rng.uniform(-0.8, 0.8, size=(self.hidden_size, 2))
        recurrent = rng.normal(0, 1, size=(self.hidden_size, self.hidden_size))
        radius = np.max(np.abs(np.linalg.eigvals(recurrent)))
        self.recurrent_weights_ = recurrent * self.spectral_radius / radius

    def _states(self, sequence, initial_state):
        state = initial_state.copy()
        states = []
        for value in sequence:
            inputs = np.array([1.0, value])
            state = np.tanh(self.input_weights_ @ inputs + self.recurrent_weights_ @ state)
            states.append(state.copy())
        return np.array(states), state

    def fit(self, sequence, targets, washout=30):
        sequence = np.asarray(sequence, dtype=float)
        targets = np.asarray(targets, dtype=float)
        self._initialize()
        states, self.state_ = self._states(sequence, np.zeros(self.hidden_size))
        design = np.column_stack([np.ones(len(states) - washout), states[washout:]])
        identity = np.eye(design.shape[1])
        identity[0, 0] = 0
        self.output_weights_ = np.linalg.solve(design.T @ design + self.regularization * identity, design.T @ targets[washout:])
        return self

    def predict(self, sequence, continue_sequence=False):
        initial = self.state_ if continue_sequence else np.zeros(self.hidden_size)
        states, final_state = self._states(np.asarray(sequence, dtype=float), initial)
        if continue_sequence:
            self.state_ = final_state
        return np.column_stack([np.ones(len(states)), states]) @ self.output_weights_


def demo():
    rng = np.random.default_rng(48)
    time = np.linspace(0, 24 * np.pi, 1000)
    series = np.sin(time) + 0.03 * rng.normal(size=len(time))
    inputs = series[:-1]
    targets = series[1:]
    split = 760
    model = RecurrentNeuralNetwork(hidden_size=70, random_state=48).fit(inputs[:split], targets[:split])
    predictions = model.predict(inputs[split:], continue_sequence=True)
    score = r2_score(targets[split:], predictions)
    assert score > 0.9
    print(f"rmse={mean_squared_error(targets[split:], predictions) ** 0.5:.3f}")
    print(f"r2={score:.3f}")


if __name__ == "__main__":
    demo()
