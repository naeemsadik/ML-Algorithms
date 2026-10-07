from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def demo():
    x, y = make_classification(
        n_samples=700,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        class_sep=1.4,
        random_state=33,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=33)
    model = RandomForestClassifier(n_estimators=180, max_features="sqrt", random_state=33, n_jobs=-1).fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.8
    print(classification_report(y_test, predictions, digits=3))
    print(f"accuracy={accuracy:.3f}")


if __name__ == "__main__":
    demo()
