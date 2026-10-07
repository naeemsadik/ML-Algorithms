from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def demo():
    x, y = make_regression(n_samples=650, n_features=10, n_informative=8, noise=12, random_state=30)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)
    model = RandomForestRegressor(n_estimators=180, max_features=0.8, random_state=30, n_jobs=-1).fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = r2_score(y_test, predictions)
    assert score > 0.8
    print(f"rmse={mean_squared_error(y_test, predictions) ** 0.5:.3f}")
    print(f"r2={score:.3f}")


if __name__ == "__main__":
    demo()
