import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

GITHUB_API_TOKEN = ""
REPO_URL = "https://api.github.com/repos/naushadnizarali/golang-learn"


def get_commits():
    url = f"{REPO_URL}/commits"
    headers = {"Authorization": f"Bearer {GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    commits = response.json()
    return commits


def get_one_commit(sha):
    url = f"{REPO_URL}/commits/{sha}"
    headers = {"Authorization": f"Bearer {GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    commit = response.json()
    return commit


def preprocess_commits(commits):
    preprocessed_commits = []
    le = LabelEncoder()

    for commit in commits:
        details = get_one_commit(commit["sha"])
        author = commit["author"]["login"]
        date = commit["commit"]["author"]["date"]
        message = commit["commit"]["message"]
        added = details["stats"]["additions"]
        deleted = details["stats"]["deletions"]
        # modified = details["stats"]["modifications"]

        # encoded_author = le.fit_transform([author])[0]
        # encoded_date = le.fit_transform([date])[0]
        # encoded_message = le.fit_transform([message])[0]
        # encoded_added = le.fit_transform([added])[0]
        # encoded_deleted = le.fit_transform([deleted])[0]

        # print("encoded_author", encoded_author)
        # print("encoded_date", encoded_date)
        # print("encoded_message", encoded_message)
        # print("encoded_added", encoded_added)
        # print("encoded_deleted", encoded_deleted)

        # preprocessed_commits.append(
        #     (
        #         encoded_author,
        #         encoded_date,
        #         encoded_message,
        #         encoded_added,
        #         encoded_deleted,
        #     )
        # )

        preprocessed_commits.append(
            (
                author,
                date,
                message,
                added,
                deleted,
            )
        )
    return preprocessed_commits


def train_model(preprocessed_commits):
    X = [
        commit[1:] for commit in preprocessed_commits
    ]  # features (date, message, added, deleted)
    y = [commit[0] for commit in preprocessed_commits]  # target (author)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_performance(model, preprocessed_commits):
    predictions = model.predict([commit[1:] for commit in preprocessed_commits])
    return predictions


def visualize_results(predictions, preprocessed_commits):
    authors = [commit[0] for commit in preprocessed_commits]
    # le = LabelEncoder()
    # decoded_authors = le.inverse_transform(authors)
    plt.bar(authors, predictions)
    plt.xlabel("Developer")
    plt.ylabel("Performance Score")
    plt.title("Developer Performance")
    plt.show()


if __name__ == "__main__":
    _commits = get_commits()
    # print(_commits[:1])

    _preprocessed_commits = preprocess_commits(_commits)
    _model = train_model(_preprocessed_commits)
    _predict_performance = predict_performance(_model, _preprocessed_commits)
    visualize_results(_predict_performance, _preprocessed_commits)
