"""
This Python file provides some useful code for reading the training file
"clean_quercus.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""
import numpy as np
import re
import pandas as pd

file_name = "clean_quercus.csv"
random_state = 42


def normalize(s, m, sd):
    """Normalize temperature and sell"""
    s = (s - m) / sd
    return float(s)


def sigmoid(x):
    """
    Apply the sigmoid activation to a numpy matrix `x` of any shape.
    """
    return 3 / (1 + np.exp(-x)) + 0.5


def pred(w, X):
    return sigmoid(np.dot(X, w.T))


def loss(w, X, t):
    y = pred(w, X)
    z = np.dot(X, w)
    return (t * np.logaddexp(np.zeros(len(y)), -z) + (1 - t) * np.logaddexp(
        np.zeros(len(y)), z)) / X.shape[0]


def grad(w, X, t):
    y = pred(w, X)
    # np.dot((y-t), X) / X.shape[0]
    return np.dot((y - t), X) / X.shape[0]


def solve_via_gradient_descent(x_train, t_train, alpha=0.0025, niter=1000):
    # initialize all the weights to zeros
    w = np.zeros(x_train.shape[1])

    for it in range(niter):
        dw = grad(w, x_train, t_train)
        w = w - alpha * dw
    return w


def normalize(s, m, sd):
    """Normalize temperature and sell"""
    s = (s - m) / sd
    return float(s)


def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)


def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]


def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """
    s = s.replace("3-D", '')
    s = s.replace("14-dimensional", '')
    n_list = get_number_list(s)
    n_list += [0] * (5 - len(n_list))
    return n_list


def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else 0


def flip(s):
    if s != [0, 0, 0, 0, 0]:
        for i in range(5):
            s[i] = 6 - s[i]


def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0


def accuracy(y, t):
    return np.mean(y == t)

def process_log(df):
    
    # Clean numerics

    df["q_sell"] = df["q_sell"].apply(to_numeric).fillna(0)
    df["q_temperature"] = df["q_temperature"].apply(to_numeric).fillna(0)
    # Normalize sell and temperature
    sells_mean = df["q_sell"].mean()
    sells_sd = df["q_sell"].std()
    temperatures_mean = df["q_temperature"].mean()
    temperatures_sd = df["q_temperature"].std()
    df["q_sell"] = df["q_sell"].apply(normalize, args=(sells_mean, sells_sd))
    df["q_temperature"] = df["q_temperature"].apply(normalize, args=(
    temperatures_mean, temperatures_sd))

    # Clean for number categories

    df["q_scary"] = df["q_scary"].apply(get_number)
    scary_mean, scary_sd = df["q_scary"].mean(), df["q_scary"].std()
    df["q_scary"].apply(normalize, args=(scary_mean, scary_sd))
    df["q_dream"] = df["q_dream"].apply(get_number)
    dream_mean, dream_sd = df["q_dream"].mean(), df["q_dream"].std()
    df["q_dream"].apply(normalize, args=(dream_mean, dream_sd))
    df["q_desktop"] = df["q_desktop"].apply(get_number)
    desktop_mean, desktop_sd = df["q_desktop"].mean(), df["q_desktop"].std()
    df["q_desktop"].apply(normalize, args=(desktop_mean, desktop_sd))

    # Create quote rank categories

    df["q_quote"] = df["q_quote"].apply(get_number_list_clean)

    df["q_quote"].apply(flip)

    # Create category indicators

    # Create multi-category indicators

    for cat in ["Parents", "Siblings", "Friends", "Teacher"]:
        df[f"q_remind_{cat}"] = df["q_remind"].apply(lambda s: cat_in_s(s, cat))
        m, sd = df[f"q_remind_{cat}"].mean(), df[f"q_remind_{cat}"].std()
        df[f"q_remind_{cat}"].apply(normalize, args=(m, sd))
    del df["q_remind"]

    for cat in ["People", "Cars", "Cats", "Fireworks", "Explosions"]:
        df[f"q_better_{cat}"] = df["q_better"].apply(lambda s: cat_in_s(s, cat))
        m, sd = df[f"q_better_{cat}"].mean(), df[f"q_better_{cat}"].std()
        df[f"q_better_{cat}"].apply(normalize, args=(m, sd))

    del df["q_better"]

    # for i in ["1","2","3","4","5"]:
    #     df[f"q_quote_{i}"] = df["q_quote"].apply(lambda s: s[int(i)-1])
    #     m, sd = df[f"q_quote_{i}"].mean(), df[f"q_quote_{i}"].std()
    #     df[f"q_quote_{i}"].apply(normalize, args=(m, sd))

    del df["q_quote"]

    df = df.drop(["q_story"], axis=1)
    return df


df = pd.read_csv(file_name)

df = process_log(df)

# Prepare data for training - use a simple train/test split for now

id = []
for i in range(0, len(df)):
    if df["user_id"][i] not in id:
        id.append(df["user_id"][i])
np.random.seed(random_state)
np.random.shuffle(id)

new_df = pd.DataFrame()
for i in range(0, len(id)):
    new_df = pd.concat([new_df, df[df["user_id"] == id[i]]])

df = new_df

x = df.drop(["label", "user_id"], axis=1).values
y = df["label"]

n_train = 500
# n_valid = 450

x_train = x[:n_train]
y_train = y[:n_train]
# # x_valid = x[n_train:n_valid]
# # y_valid = y[n_train:n_valid]
# x_test = x[n_train:]
# y_test = y[n_train:]

# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(fit_intercept=False)

# lr.fit(x_train, y_train)

# # print(lr.predict(x_train))
# print(f"Training accuracy: {lr.score(x_train, y_train)}")
# # print(f"Validation accuracy: {lr.score(x_valid, y_valid)}")
# print(f"Test accuracy: {lr.score(x_test, y_test)}")

w = solve_via_gradient_descent(x_train, y_train, alpha=0.005, niter=5000)
class LogisticReg:
    def __init__(self, w) -> None:
        self.w = w
    def predict(self, x):
        return np.around(pred(w, x))
model = LogisticReg(w)
    # y_train_p = np.around(pred(w, x_train))
    # # y_valid_p = np.around(pred(w, x_valid))
    # y_test_p = np.around(pred(w, x_test))

    # print(accuracy(y_train_p, y_train))
    # # print(accuracy(y_valid_p, y_valid))
    # print(accuracy(y_test_p, y_test))

    # Train and evaluate classifiers
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf.fit(x_train, y_train)
    # train_acc = clf.score(x_train, y_train)
    # test_acc = clf.score(x_test, y_test)
    # print(f"{type(clf).__name__} train acc: {train_acc}")
    # print(f"{type(clf).__name__} test acc: {test_acc}")
