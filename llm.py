import re
import numpy as np


class TransformerLanguageModel:
    def __init__(self, context_size=6, dimensions=32, learning_rate=0.4, epochs=500, random_state=42):
        self.context_size = context_size
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b|[.,!?]", text.lower())

    def _softmax(self, values):
        shifted = values - np.max(values, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    def _normalize(self, values):
        return (values - values.mean(axis=-1, keepdims=True)) / np.sqrt(values.var(axis=-1, keepdims=True) + 1e-6)

    def _encode(self, token_ids):
        hidden = self.embeddings_[token_ids] + self.positions_
        queries = hidden @ self.query_
        keys = hidden @ self.key_
        values = hidden @ self.value_
        scores = queries @ keys.T / np.sqrt(self.dimensions)
        scores[np.triu_indices(self.context_size, 1)] = -1e9
        attention = self._softmax(scores)
        hidden = self._normalize(hidden + attention @ values @ self.attention_output_)
        feed_forward = np.maximum(0, hidden @ self.feed_forward_in_) @ self.feed_forward_out_
        return self._normalize(hidden + feed_forward)[-1]

    def _context(self, token_ids):
        values = token_ids[-self.context_size:]
        return [self.pad_id_] * (self.context_size - len(values)) + values

    def fit(self, text):
        tokens = self._tokenize(text)
        vocabulary = sorted(set(tokens))
        self.id_to_token_ = ["<pad>", "<unk>"] + vocabulary
        self.token_to_id_ = {token: index for index, token in enumerate(self.id_to_token_)}
        self.pad_id_ = self.token_to_id_["<pad>"]
        ids = [self.token_to_id_[token] for token in tokens]
        rng = np.random.default_rng(self.random_state)
        scale = 1 / np.sqrt(self.dimensions)
        size = len(self.id_to_token_)
        self.embeddings_ = rng.normal(0, scale, size=(size, self.dimensions))
        self.embeddings_[self.pad_id_] = 0
        self.positions_ = rng.normal(0, scale, size=(self.context_size, self.dimensions))
        self.query_ = rng.normal(0, scale, size=(self.dimensions, self.dimensions))
        self.key_ = rng.normal(0, scale, size=(self.dimensions, self.dimensions))
        self.value_ = rng.normal(0, scale, size=(self.dimensions, self.dimensions))
        self.attention_output_ = rng.normal(0, scale, size=(self.dimensions, self.dimensions))
        self.feed_forward_in_ = rng.normal(0, scale, size=(self.dimensions, self.dimensions * 2))
        self.feed_forward_out_ = rng.normal(0, scale, size=(self.dimensions * 2, self.dimensions))
        contexts = [self._context(ids[:index]) for index in range(1, len(ids))]
        targets = np.array(ids[1:])
        features = np.array([self._encode(context) for context in contexts])
        self.output_weights_ = rng.normal(0, scale, size=(self.dimensions, size))
        self.output_bias_ = np.zeros(size)
        self.loss_curve_ = []
        rows = np.arange(len(targets))
        for epoch in range(self.epochs):
            probabilities = self._softmax(features @ self.output_weights_ + self.output_bias_)
            if epoch in (0, self.epochs - 1):
                self.loss_curve_.append(float(-np.log(probabilities[rows, targets] + 1e-12).mean()))
            gradient = probabilities
            gradient[rows, targets] -= 1
            gradient /= len(targets)
            self.output_weights_ -= self.learning_rate * features.T @ gradient
            self.output_bias_ -= self.learning_rate * gradient.sum(axis=0)
        return self

    def next_token_probabilities(self, text):
        ids = [self.token_to_id_.get(token, self.token_to_id_["<unk>"]) for token in self._tokenize(text)]
        feature = self._encode(self._context(ids))
        return self._softmax(feature @ self.output_weights_ + self.output_bias_)

    def generate(self, prompt, max_new_tokens=12):
        generated = self._tokenize(prompt)
        for _ in range(max_new_tokens):
            probabilities = self.next_token_probabilities(" ".join(generated))
            token = self.id_to_token_[int(np.argmax(probabilities))]
            generated.append(token)
        text = " ".join(generated)
        return re.sub(r"\s+([.,!?])", r"\1", text)


def demo():
    passage = " ".join([
        "machine learning finds patterns in data. supervised learning uses labeled examples. unsupervised learning discovers hidden structure. neural networks learn useful representations."
        for _ in range(24)
    ])
    model = TransformerLanguageModel(random_state=60).fit(passage)
    generated = model.generate("machine learning", max_new_tokens=10)
    assert model.loss_curve_[-1] < model.loss_curve_[0]
    assert generated.startswith("machine learning finds patterns")
    print(f"initial_loss={model.loss_curve_[0]:.3f}")
    print(f"final_loss={model.loss_curve_[-1]:.3f}")
    print(generated)


if __name__ == "__main__":
    demo()
