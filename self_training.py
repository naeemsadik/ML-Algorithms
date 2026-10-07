import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier


def demo():
    x, y = make_classification(
        n_samples=700,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        class_sep=1.3,
        random_state=27,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=27)
    rng = np.random.default_rng(27)
    partially_labeled = y_train.copy()
    partially_labeled[rng.random(len(y_train)) < 0.8] = -1
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=27))
    model = SelfTrainingClassifier(estimator, threshold=0.8, max_iter=20).fit(x_train, partially_labeled)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.8
    assert np.count_nonzero(model.transduction_ != -1) > np.count_nonzero(partially_labeled != -1)
    print(f"initial_labels={np.count_nonzero(partially_labeled != -1)}")
    print(f"final_labels={np.count_nonzero(model.transduction_ != -1)}")
    print(f"accuracy={accuracy:.3f}")


if __name__ == "__main__":
    demo()
