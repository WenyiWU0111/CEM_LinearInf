import pandas as pd
import numpy as np

def data_generation(n: int = 1000,
                    p: float = 0.3,
                    att: int = 3,
                    x_cont: list = [0, 1, 5],
                    x_cate: list = [3],
                    con_x: list = [(0, 1), (2, -0.5), (5, 3)],
                    x_weights: list = None):

    """
    Method for data simulation, return a dataframe with control X, treatment T, and result Y.

    Parameters
    ----------

    n: integer
        number of samples.

    p: float
        propotion of samples being treated.

    att: float
        average treatment effect on treated.

    x_cont: list = [mean, std, dimension]
        Continuous X variables are generated with normal distribution. Three numbers in the x_cont list refer to the
        mean, standard deviation of the normal distribution, and the number of continuous X variables respectively.

    x_cate: list
        A list of integers referring how many categories each categorical variable will have.

    con_x: list
        A list of tuples specifing which X variables are the confounders, and their effects on T.
        For example, (0, 1) means the first variable X is a confounder, and it has 1 unit of effect on T.

    x_weights: list
        A list of floats specifing the weight of each variable X when generating Y, and the length of the list must be equal to the
        dimension of X variables. If x_weights is not specified, it will be generated with uniform distribution.

    """

    mean = x_cont[0]
    sigma = x_cont[1]
    x_cont_dim = x_cont[2]
    cont_x = np.random.normal(mean, sigma, size=(n, x_cont_dim))

    cate_list = []
    for cate in x_cate:
        cate_x_i = np.random.randint(cate, size=n)
        cate_list.append(cate_x_i.reshape(-1, 1))

    if len(cate_list) > 1:
        cate_x = np.concatenate(cate_list, axis=1)
        X = np.concatenate([cont_x, cate_x], axis=1)
    elif len(cate_list) == 1:
        cate_x = cate_list[0].reshape(-1, 1)
        X = np.concatenate([cont_x, cate_x], axis=1)
    else:
        raise ValueError("The parameter 'x_cate' for categorical X cannot be an empty list!")

    X_dim = x_cont_dim + len(cate_list)
    X_names = [f"X{i}" for i in range(1, X_dim + 1)]
    df = pd.DataFrame(X, columns=X_names)

    noise_X = np.random.randn(n).reshape(-1, 1)

    linear_pred = noise_X
    for params in con_x:
        column = params[0]
        weight = params[1]
        linear_pred += weight * X[:, column].reshape(-1, 1)

    pred_p = p * np.exp(linear_pred + 1e-5) / (1 + np.exp(linear_pred + 1e-5))
    T = np.random.binomial(1, pred_p)
    #df['pred_p'] = pred_p
    df['T'] = T

    noise_Y = np.random.randn(n).reshape(-1, 1)
    if x_weights == None:
        x_weights = 10 * np.random.rand(X_dim) - 5
    elif len(x_weights) != X_dim:
        raise ValueError("The length of the 'x_weight' list should equal the dimension of X!")

    Y = 1 + att * T + noise_Y
    for i in range(X_dim):
        Y += x_weights[i] * X[:, i].reshape(-1, 1)

    df['Y'] = Y

    return df