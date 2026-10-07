import numpy as np


class HiddenMarkovModel:
    def __init__(self, start_probabilities, transition_probabilities, emission_probabilities, random_state=42):
        self.start_probabilities = np.asarray(start_probabilities, dtype=float)
        self.transition_probabilities = np.asarray(transition_probabilities, dtype=float)
        self.emission_probabilities = np.asarray(emission_probabilities, dtype=float)
        self.random_state = random_state

    def sample(self, length):
        rng = np.random.default_rng(self.random_state)
        states = np.empty(length, dtype=int)
        observations = np.empty(length, dtype=int)
        states[0] = rng.choice(len(self.start_probabilities), p=self.start_probabilities)
        observations[0] = rng.choice(self.emission_probabilities.shape[1], p=self.emission_probabilities[states[0]])
        for step in range(1, length):
            states[step] = rng.choice(len(self.start_probabilities), p=self.transition_probabilities[states[step - 1]])
            observations[step] = rng.choice(self.emission_probabilities.shape[1], p=self.emission_probabilities[states[step]])
        return observations, states

    def forward(self, observations):
        observations = np.asarray(observations, dtype=int)
        probabilities = self.start_probabilities * self.emission_probabilities[:, observations[0]]
        scale = probabilities.sum()
        probabilities /= scale
        log_likelihood = np.log(scale)
        for observation in observations[1:]:
            probabilities = probabilities @ self.transition_probabilities * self.emission_probabilities[:, observation]
            scale = probabilities.sum()
            probabilities /= scale
            log_likelihood += np.log(scale)
        return float(log_likelihood)

    def viterbi(self, observations):
        observations = np.asarray(observations, dtype=int)
        log_start = np.log(self.start_probabilities)
        log_transition = np.log(self.transition_probabilities)
        log_emission = np.log(self.emission_probabilities)
        scores = log_start + log_emission[:, observations[0]]
        paths = np.empty((len(observations), len(scores)), dtype=int)
        paths[0] = -1
        for step, observation in enumerate(observations[1:], start=1):
            candidates = scores[:, None] + log_transition
            paths[step] = np.argmax(candidates, axis=0)
            scores = np.max(candidates, axis=0) + log_emission[:, observation]
        states = np.empty(len(observations), dtype=int)
        states[-1] = np.argmax(scores)
        for step in range(len(observations) - 1, 0, -1):
            states[step - 1] = paths[step, states[step]]
        return states


def demo():
    model = HiddenMarkovModel(
        start_probabilities=[0.5, 0.5],
        transition_probabilities=[[0.88, 0.12], [0.15, 0.85]],
        emission_probabilities=[[0.75, 0.2, 0.05], [0.08, 0.27, 0.65]],
        random_state=54,
    )
    observations, states = model.sample(160)
    decoded = model.viterbi(observations)
    accuracy = np.mean(decoded == states)
    likelihood = model.forward(observations)
    assert accuracy > 0.8
    assert np.isfinite(likelihood)
    print(f"log_likelihood={likelihood:.3f}")
    print(f"state_accuracy={accuracy:.3f}")


if __name__ == "__main__":
    demo()
