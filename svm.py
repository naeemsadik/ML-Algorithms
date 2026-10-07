from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def demo():
    x, y = make_circles(n_samples=700, factor=0.4, noise=0.08, random_state=57)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=57)
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3.0, gamma="scale")).fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.95
    print(classification_report(y_test, predictions, digits=3))
    print(f"accuracy={accuracy:.3f}")


if __name__ == "__main__":
    demo()
