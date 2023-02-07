"""
This Python file provides some useful code for reading the training file
"clean_quercus.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""
import random
import re
import pandas as pd
import numpy as np
import string

file_name = "clean_quercus.csv"
random_state = 42
pi_mle = None
theta_mle = None


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
    n_list += [-1] * (5 - len(n_list))
    return n_list


def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1


def find_quote_at_rank(l, i):
    """Return the quote at a certain rank in list `l`.

    Quotes are indexed starting at 1 as ordered in the survey.

    If quote is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1


def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0


def make_vocab(data):
    vocab, vocab_occ = [], []
    for i in range(data.shape[0]):
        line = data[i, -1]
        if type(line) is not float:
            for pun in string.punctuation:
                line = line.replace(pun, "")
            line = line.replace("\xa0", "").replace("\n", "").lower()
            data[i, -1] = line
            for word in line.split(" "):
                if word not in vocab:
                    vocab.append(word)
                    vocab_occ.append(1)
                else:
                    vocab_occ[vocab.index(word)] += 1
    # vocab_occ = np.array(vocab_occ)
    # filted_vocab = []
    # for i in range(len(vocab)):
    #     if vocab_occ[i] >= 5:
    #         filted_vocab.append(vocab[i])
    # return filted_vocab, data
    return vocab, data


def make_bow(data, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
        `t`: A numpy array of shape [len(data)], with `t[i] == 1` if
             `data[i]` is a positive review, and `t[i] == 0` otherwise.
    """
    X = np.zeros([len(data), len(vocab)])
    i = 0
    for review in data[:, -1]:
        if type(review) is not float:
            for word in review.split(" "):
                if word in vocab:
                    X[i][vocab.index(word)] = 1
        i += 1
    data = np.delete(data, obj=-1, axis=1)
    data = np.delete(data, obj=-1, axis=1)
    data = np.delete(data, obj=-1, axis=1)
    data = np.concatenate((data, X), axis=1)
    return data.astype(int)
    # return X


def naive_bayes_mle(X, t):
    """
    Compute the parameters $pi$ and $theta_{jc}$ that maximizes the log-likelihood
    of the provided data (X, t).

    **Your solution should be vectorized, and contain no loops**

    Parameters:
        `X` - a matrix of bag-of-word features of shape [N, V],
              where N is the number of data points and V is the vocabulary size.
              X[i,j] should be either 0 or 1. Produced by the make_bow() function.
        `t` - a vector of class labels of shape [N], with t[i] being either 0 or 1.
              Produced by the make_bow() function.

    Returns:
        `pi` - a scalar; the MLE estimate of the parameter $\pi = p(c = 1)$
        `theta` - a matrix of shape [V, 2], where `theta[j, c]` corresponds to
                  the MLE estimate of the parameter $\theta_{jc} = p(x_j = 1 | c)$
    """
    N, vocab_size = X.shape[0], X.shape[1]
    pi = [1 / 3, 1 / 3, 1 / 3]

    # these matrices may be useful (but what do they represent?)
    t = np.array(t)
    X_1 = X[t[:, 0] == 1]
    X_2 = X[t[:, 1] == 1]
    X_3 = X[t[:, 2] == 1]

    # theta[:, 1] = None # you may uncomment this line if you'd like
    # theta[:, 0] = None # you may uncomment this line if you'd like
    theta_1 = np.sum(X_1, axis=0) / X_1.shape[0]
    theta_2 = np.sum(X_2, axis=0) / X_2.shape[0]
    theta_3 = np.sum(X_3, axis=0) / X_3.shape[0]
    theta = np.array([theta_1, theta_2, theta_3]).T

    return pi, theta


def naive_bayes_map(X, t, alpha, beta):
    """
    Compute the parameters $pi$ and $theta_{jc}$ that maximizes the posterior
    of the provided data (X, t). We will use the beta distribution with
    $a=2$ and $b=2$ for all of our parameters.

    **Your solution should be vectorized, and contain no loops**

    Parameters:
        `X` - a matrix of bag-of-word features of shape [N, V],
              where N is the number of data points and V is the vocabulary size.
              X[i,j] should be either 0 or 1. Produced by the make_bow() function.
        `t` - a vector of class labels of shape [N], with t[i] being either 0 or 1.
              Produced by the make_bow() function.

    Returns:
        `pi` - a scalar; the MAP estimate of the parameter $\pi = p(c = 1)$
        `theta` - a matrix of shape [V, 2], where `theta[j, c]` corresponds to
                  the MAP estimate of the parameter $\theta_{jc} = p(x_j = 1 | c)$
    """
    N, vocab_size = X.shape[0], X.shape[1]

    t = np.array(t)
    X_1 = X[t[:, 0] == 1]
    X_2 = X[t[:, 1] == 1]
    X_3 = X[t[:, 2] == 1]
    pi = [(len(X_1) + alpha - 1) / (N + alpha + beta - 2),
          (len(X_2) + alpha - 1) / (N + alpha + beta - 2),
          (len(X_3) + alpha - 1) / (N + alpha + beta - 2)]

    theta_1 = (np.sum(X_1, axis=0) + alpha - 1) / (
            X_1.shape[0] + alpha + beta - 2)
    theta_2 = (np.sum(X_2, axis=0) + alpha - 1) / (
            X_2.shape[0] + alpha + beta - 2)
    theta_3 = (np.sum(X_3, axis=0) + alpha - 1) / (
            X_3.shape[0] + alpha + beta - 2)

    theta = np.array([theta_1, theta_2, theta_3]).T

    return pi, theta


def make_prediction(X, pi, theta):
    # X.shape = [N, V]
    # pi = Constant
    # theta.shape = [V, 2]
    negate_X = np.ones(X.shape)
    negate_X[X == 1] = 0
    one_pred = X * theta.T[0] + negate_X * (1 - theta.T[0])
    two_pred = X * theta.T[1] + negate_X * (1 - theta.T[1])
    three_pred = X * theta.T[2] + negate_X * (1 - theta.T[2])
    one_pred[one_pred == 0] = 0.00001
    two_pred[two_pred == 0] = 0.00001
    three_pred[three_pred == 0] = 0.00001
    one_pred = np.exp(np.sum(np.log(one_pred), axis=1)) * pi[0]
    two_pred = np.exp(np.sum(np.log(two_pred), axis=1)) * pi[1]
    three_pred = np.exp(np.sum(np.log(three_pred), axis=1)) * pi[2]
    y = []
    for i in range(len(one_pred)):
        max_p = max(one_pred[i], two_pred[i], three_pred[i])
        if max_p == one_pred[i]:
            y.append(1)
        elif max_p == two_pred[i]:
            y.append(2)
        else:
            y.append(3)
    return np.array(y)


def accuracy(y, t):
    return np.mean(y == t)


df = pd.read_csv(file_name)
# Clean numerics
df["q_sell"] = df["q_sell"].apply(to_numeric).fillna(0)
df["q_temperature"] = df["q_temperature"].apply(to_numeric).fillna(0)

# # Get sells and temperatures
# sells = df["q_sell"].apply(to_numeric).fillna(0)
# temperatures = df["q_temperature"].apply(to_numeric).fillna(0)
sells_mean = df["q_sell"].mean()
sells_sd = df["q_sell"].std()
temperatures_mean = df["q_temperature"].mean()
temperatures_sd = df["q_temperature"].std()

df["q_sell"] = df["q_sell"].apply(normalize, args=(sells_mean, sells_sd))
df["q_temperature"] = df["q_temperature"].apply(normalize, args=(
    temperatures_mean, temperatures_sd))

# print(df["q_temperature"][174])
# Clean for number categories

df["q_scary"] = df["q_scary"].apply(get_number)
df["q_dream"] = df["q_dream"].apply(get_number)
df["q_desktop"] = df["q_desktop"].apply(get_number)

# Create quote rank categories

df["q_quote"] = df["q_quote"].apply(get_number_list_clean)

temp_names = []
for i in range(1, 6):
    col_name = f"rank_{i}"
    temp_names.append(col_name)
    df[col_name] = df["q_quote"].apply(lambda l: find_quote_at_rank(l, i))
del df["q_quote"]

# Create category indicators

new_names = []
for col in ["q_scary"] + ["q_dream"] + ["q_desktop"] + temp_names:
    indicators = pd.get_dummies(df[col], prefix=col)
    new_names.extend(indicators.columns)
    df = pd.concat([df, indicators], axis=1)
    del df[col]

# Create multi-category indicators

for cat in ["Parents", "Siblings", "Friends", "Teacher"]:
    df[f"q_remind_{cat}"] = df["q_remind"].apply(lambda s: cat_in_s(s, cat))
    new_names.append(f"q_remind_{cat}")

del df["q_remind"]

for cat in ["People", "Cars", "Cats", "Fireworks", "Explosions"]:
    df[f"q_better_{cat}"] = df["q_better"].apply(lambda s: cat_in_s(s, cat))
    new_names.append(f"q_better_{cat}")

del df["q_better"]

# Prepare data for training - use a simple train/test split for now
df = df[new_names + ["q_sell", "q_temperature", "label", "user_id", "q_story"]]

# Split data into train and test sets same id should in same set
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
y = pd.get_dummies(df["label"].values)
i_train = 300
i_valid = 450


x_train = x
y_train = y
# x_valid = x[i_train:i_valid]
# y_valid = y[i_train:i_valid]
# x_test = x[550:]
# y_test = y[550:]

# Train and evaluate classifiers
vocabs, x_train = make_vocab(x_train)
# v_valid, x_valid = make_vocab(x_valid)
# vocabs.extend(v_valid)
# v, x_test = make_vocab(x_test)
x_train = make_bow(x_train, vocabs)
# x_valid = make_bow(x_valid, vocabs)
# x_test = make_bow(x_test, vocabs)
# pi_mle, theta_mle = naive_bayes_mle(x_train, y_train)
pi_map, theta_map = naive_bayes_map(x_train, y_train, alpha=10, beta=10)


class NaiveBayes:
    def __init__(self, pi_map, theta_map) -> None:
        self.pi_map = pi_map
        self.theta_map = theta_map

    def predict(self, x):
        x = x.reshape(1, x.shape[0])
        v, x = make_vocab(x)
        x = make_bow(x, vocabs)
        return make_prediction(x, self.pi_map, self.theta_map)


model = NaiveBayes(pi_map, theta_map)

# Predict and report accuracy
# y_mle_train = make_prediction(x_train, pi_mle, theta_mle)
# y_mle_valid = make_prediction(x_valid, pi_mle, theta_mle)
# y_mle_test = make_prediction(x_test, pi_mle, theta_mle)
# y_map_train = make_prediction(x_train, pi_map, theta_map)
# y_map_valid = make_prediction(x_valid, pi_map, theta_map)
# y_map_test = make_prediction(x_test, pi_map, theta_map)
# y2 = np.array(y_train)
# y_train = np.ones(y2.shape[0])
# y_train[y2[:, 1] == 1] = 2
# y_train[y2[:, 1] == 2] = 3
# y_valid2 = y_valid
# y2 = np.array(y_valid)
# y_valid = np.ones(y2.shape[0])
# y_valid[y2[:, 1] == 1] = 2
# y_valid[y2[:, 1] == 2] = 3
# y2 = np.array(y_test)
# y_test = np.ones(y2.shape[0])
# y_test[y2[:, 1] == 1] = 2
# y_test[y2[:, 1] == 2] = 3
# print("MLE Train Acc:", accuracy(y_mle_train, y_train))
# print("MLE Valid Acc:", accuracy(y_mle_valid, y_valid))
# print("MLE Test Acc:", accuracy(y_mle_test, y_test))
# print("MAP Train Acc:", accuracy(y_map_train, y_train))
# print("MAP Valid Acc:", accuracy(y_map_valid, y_valid))
# print("MAP Test Acc:", accuracy(y_map_test, y_test))

