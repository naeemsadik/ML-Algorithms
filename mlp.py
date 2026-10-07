from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def demo():
    x, y = make_moons(n_samples=700, noise=0.18, random_state=45)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=45)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs", max_iter=1500, random_state=45),
    ).fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.9
    print(classification_report(y_test, predictions, digits=3))
    print(f"accuracy={accuracy:.3f}")


if __name__ == "__main__":
    demo()
