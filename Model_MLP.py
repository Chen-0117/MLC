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

file_name = "clean_quercus.csv"
random_state = 42

def softmax(z):
    """    
    Compute the softmax of vector z, or row-wise for a matrix z.    
    We subtract z.max(axis=1) from each row for numerical stability.    
    Parameters: `z` - a numpy array of shape (K,) or (N, K)    
    Returns: a numpy array with the same shape as `z`, with the softmax        
    activation applied to each row of `z`    
    """
    m = z.max(axis=1, keepdims = True)
    y = np.exp(z-m)/np.sum(np.exp(z-m), axis = 1, keepdims = True)
    return y

class MLPModel(object):
    def __init__(self, num_features=64*1, num_hidden=100, num_classes=3):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.weights1 = np.zeros([num_features, num_hidden])
        self.bias1 = np.zeros([num_hidden])
        self.weights2 = np.zeros([num_hidden, num_classes])
        self.bias2 = np.zeros([num_classes])
        self.cleanup()
        self.initializeParams()

    def initializeParams(self):
        self.weights1 = np.random.normal(0, 2/self.num_features, self.
        weights1.shape)
        self.bias1 = np.random.normal(0, 2/self.num_features, self.bias1.shape)
        self.weights2 = np.random.normal(0, 2/self.num_hidden, self.weights2.shape)
        self.bias2 = np.random.normal(0, 2/self.num_hidden, self.bias2.shape)

    def forward(self, X):
        """
        Compute the forward pass to produce prediction for inputs.
        This function also keeps some of the intermediate values in
        the neural network computation, to make computing gradients ea
        sier.
        For the ReLU activation, you may find the function `np.maximum
        ` helpful
        Parameters:
        `X` - A numpy array of shape (N, self.num_features)
        Returns: A numpy array of predictions of shape (N, self.num_cl
        asses)
        """
        self.N = X.shape[0]
        self.X = X
        self.z1 = np.matmul(self.X, self.weights1) + np.zeros([self.N,self.num_hidden]) # the hidden state value (pre-activation)
        self.h = np.maximum(np.zeros(self.z1.shape), self.z1) # the hidden state value (post ReLU activation)
        self.z2 = np.matmul(self.h, self.weights2) + np.zeros([self.N,self.num_classes]) # the logit scores (pre-activation)
        # self.y = np.maximum(np.zeros(self.z2.shape), self.z2) # the class probabilities (post ReLU activation)
        self.y = softmax(self.z2) # the class probabilities (post Softmax activation)
        return self.y

    def backward(self, ts):
        """
        Compute the backward pass, given the ground-truth, one-hot tar
        gets.
        You may assume that the `forward()` method has been called for
        the
        corresponding input `X`, so that the quantities computed in th
        e
        `forward()` method is accessible.
        The member variables you store here will be used in the `updat
        e()`
        method. The shape of each error signal should be the same as t
        he shape
        of the corresponding forward pass quantity,
        i.e. the shape of `self.z2_bar` should be the same as `self.z2
        `,
        the shape of `self.w2_bar` should be the same as `self.we
        ights2`, etc
        Parameters:
        `ts` - A numpy array of shape (N, self.num_classes)
        """
        self.y_bar = (self.y-ts)/ts.shape[0]
        # self.z2_bar = (self.y-ts)/ ts.shape[0] 
        # self.z2_bar = np.where(self.z2 > 0, self.y_bar, 0) 
        self.z2_bar = self.y_bar
        self.w2_bar = np.matmul(self.h.T, self.z2_bar)
        self.b2_bar = self.z2_bar*1 
        self.h_bar = np.matmul(self.z2_bar,self.weights2.T) 
        # self.z1_bar = self.h_bar*1 
        self.z1_bar = np.where(self.z1 > 0, self.h_bar, 0)
        self.w1_bar = np.matmul(self.X.T, self.z1_bar) 
        self.b1_bar = self.z1_bar*1

    def predict(self, X):
        X = np.delete(X, -1, axis=0)
        X = X.astype(float)
        self.cleanup()
        y = self.forward(X)
        il = np.zeros(y.shape[0])
        for i in range(self.y.shape[0]):
            line = y[i]
            if line[0] >= line[1]:
                if line[0] >= line[2]:
                    il[i] = 1
                else:
                    il[i] = 2
            else:
                if line[1] >= line[2]:
                    il[i] = 2
                else:
                    il[i] = 3
        return il

    # def predict(self, X):
    #     X = np.delete(X, -1, axis=1)
    #     X = X.astype(float)
    #     y = self.forward(X)
    #     il = np.zeros(y.shape)
    #     for i in range(self.y.shape[0]):
    #         line = y[i]
    #         mi = -1
    #         if line[0] >= line[1]:
    #             if line[0] >= line[2]:
    #                 mi = 0
    #             else:
    #                 mi = 2
    #         else:
    #             if line[1] >= line[2]:
    #                 mi = 1
    #             else:
    #                 mi = 2
    #         il[i][mi] = 1
    #     return il

    def update(self, alpha):
        """
        Compute the gradient descent update for the parameters of this
        model.
        Parameters:
        `alpha` - A number representing the learning rate
        """
        self.weights1 = self.weights1 - alpha * self.w1_bar
        self.bias1 = self.bias1 - alpha * self.b1_bar
        self.weights2 = self.weights2 - alpha * self.w2_bar
        self.bias2 = self.bias2 - alpha * self.b2_bar
    
    def cleanup(self):
        self.N = None
        self.X = None
        self.z1 = None
        self.h = None
        self.z2 = None
        self.y = None
        self.z2_bar = None
        self.w2_bar = None
        self.b2_bar = None
        self.h_bar = None
        self.z1_bar = None
        self.w1_bar = None
        self.b1_bar = None

def train_sgd(model, X_train, t_train,
    alpha=0.1, n_epochs=0, batch_size=100,
    X_valid=None, t_valid=None,
    w_init=None, plot=True):
    '''
    Given `model` - an instance of MLPModel
    `X_train` - the data matrix to use for training
    `t_train` - the target vector to use for training
    `alpha` - the learning rate.
    From our experiments, it appears that a larger lea
    rning rate
    is appropriate for this task.
    `n_epochs` - the number of **epochs** of gradient descent to
    run
    `batch_size` - the size of each mini batch
    `X_valid` - the data matrix to use for validation (optional)
    `t_valid` - the target vector to use for validation (optiona
    l)
    `w_init` - the initial `w` vector (if `None`, use a vector o
    f all zeros)
    `plot` - whether to track statistics and plot the training c
    urve
    Solves for logistic regression weights via stochastic gradient des
    cent,
    using the provided batch_size.
    Return weights after `niter` iterations.
    '''
    # as before, initialize all the weights to zeros
    X_train = np.delete(X_train, -1, axis=1)
    X_train = X_train.astype(float)
    w = np.zeros(X_train.shape[1])
    # track the number of iterations
    niter = 0
    # we will use these indices to help shuffle X_train
    N = X_train.shape[0] # number of training data points
    indices = list(range(N))
    for e in range(n_epochs):
        random.shuffle(indices) # for creating new minibatches
        for i in range(0, N, batch_size):
            if (i + batch_size) > N:
            # At the very end of an epoch, if there are not enough
            # data points to form an entire batch, then skip this
                continue
            indices_in_batch = np.array(indices[i: i+batch_size])
            t_train = np.array(t_train)
            X_minibatch = X_train[indices_in_batch, :]
            t_minibatch = t_train[indices_in_batch]
            
            # gradient descent iteration
            model.cleanup()
            model.forward(X_minibatch)
            model.backward(t_minibatch)
            model.update(alpha)
            niter+=1

def acc(model, x, t):
    y = model.predict(x)
    t = np.array(t)
    n = 0
    for i in range(len(t)):
        if (y[i] == t[i]).all():
            n += 1
    return n/len(t)

def normalize(s, m, sd):
    """Normalize temperature and sell"""
    s = (s - m)/sd    
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
    n_list += [-1]*(5-len(n_list))
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

def process_data(df):
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
    df["q_temperature"] = df["q_temperature"].apply(normalize, args=(temperatures_mean, temperatures_sd))
    # print(df["q_temperature"][174])
    # Clean for number categories

    df["q_scary"] = df["q_scary"].apply(get_number)
    df["q_dream"] = df["q_dream"].apply(get_number)
    df["q_desktop"] = df["q_desktop"].apply(get_number)


    # Create quote rank categories

    df["q_quote"] = df["q_quote"].apply(get_number_list_clean)

    temp_names = []
    for i in range(1,6):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        df[col_name] = df["q_quote"].apply(lambda l: find_quote_at_rank(l, i))
    del df["q_quote"]

    # Create category indicators

    new_names = []
    for col in ["q_scary"] + ["q_dream"] + ["q_desktop"] + temp_names:
        indicators = pd.get_dummies(df[col], prefix=col)
        if (col == "q_scary"):
            for i in list(set([-1,1,2,3,4,5,6,7,8,9,10]).symmetric_difference(df[col].unique())):
                col_name = col + f"_{i}"
                indicators[col_name] = 0
        else:
            for i in list(set([-1,1,2,3,4,5]).symmetric_difference(df[col].unique())):
                col_name = col + f"_{i}"
                indicators[col_name] = 0
        new_names.extend(indicators.columns)
        df = pd.concat([df, indicators], axis=1)
        del df[col]

    # Create multi-category indicators

    for cat in ["Parents", "Siblings", "Friends", "Teacher"]:
        df[f"q_remind_{cat}"] = df["q_remind"].apply(lambda s: cat_in_s(s, cat))
        new_names.append(f"q_remind_{cat}")

    del df["q_remind"]


    for cat in ["People","Cars","Cats","Fireworks","Explosions"]:
        df[f"q_better_{cat}"] = df["q_better"].apply(lambda s: cat_in_s(s, cat))
        new_names.append(f"q_better_{cat}")

    del df["q_better"]
    # Prepare data for training - use a simple train/test split for now
    if ("label" in df.keys()):
        df = df[new_names + ["q_sell", "q_temperature", "label", "user_id", "q_story"]]
    else:
        df = df[new_names + ["q_sell", "q_temperature", "user_id", "q_story"]]
    return df

df = pd.read_csv(file_name)

df = process_data(df)

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

# df = df.sample(frac=1, random_state=random_state)
# df.groupby("user_id").sample(frac=1, random_state=random_state)
# df.groupby('user_id', group_keys=False).apply(lambda x: x.sample(3))
# df = df.groupby("user_id").apply(lambda x: x.sample(frac=1, random_state=random_state))
# df = df.reset_index(drop=True)

x = df.drop(["label", "user_id"], axis=1).values
y = pd.get_dummies(df["label"].values)

# x = df.drop("user_id", axis=1).values
n_train = 500
# n_valid = 150

# id = []
# for i in range(0, len(df)):
#     if df["user_id"][i] not in id:
#         id.append(df["user_id"][i])
# np.random.seed(random_state)
# np.random.shuffle(id)

# new_df = pd.DataFrame()
# for i in range(0, len(id)):
#     if i < len(id):
#         new_df = new_df.append(df[df["user_id"] == id[i]])
#     else:
#         new_df = new_df.append(df[df["user_id"] == id[i]])


x_train = x[:n_train]
y_train = y[:n_train]

# x_valid = x[n_train:n_train+n_valid]
# y_valid = y[n_train:n_train+n_valid]

# x_test = x[n_train+n_valid:]
# y_test = y[n_train+n_valid:]
x_test = x[n_train:]
y_test = y[n_train:]

# Train and evaluate classifiers


model11 = MLPModel()
train_sgd(model11, x_train, y_train, alpha=0.01, batch_size=100, n_epochs=500)
# print('mlp training accuracy:', acc(model11, x_train, y_train))
# print('mlp accuracy:', acc(model11, x, y))



