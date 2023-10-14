import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
import statsmodels.api as sm

class inference():
    """
    Class of linear inference methods for estimating average treatment effect and heterogeneous treatment effect.

    After conducting the coarsened exact matching and imbalance checking, we can estimate the average treatment effect ATT
    and heterogeneous treatment effect HTE with statistical inference methods.

    Ordinal least square linear regression and weighted least square linear regression methods are provided for the
    ATT estimation, and the linear double machine learning method is provided for the HTE estimation.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe after matching.

    col_y: string
        The column name of dependent variable Y in your dataframe. If not specified, it would be "Y".

    col_t: string
        The column name of treatment variable T in your dataframe. If not specified, it would be "T".

    col_x: list
        A list of column names of control variables X in your dataframe, which must be specified.
        Names of confounders should not be included in this list.

    confounder_cols: list
        A list of column names of confounders W in your dataframe, which must be specified.

    """

    def __init__(self, df: pd.DataFrame, col_y: str = 'Y', col_t: str = 'T', col_x: list = None,
                 confounder_cols: list = None):

        self.df = df
        self.col_y = col_y
        self.col_t = col_t
        self.col_x = col_x
        self.confounder_cols = confounder_cols

        if col_y not in df.columns:
            raise ValueError("Please input the correct column name of y!")
        if col_t not in df.columns:
            raise ValueError("Please input the correct column name of t!")
        if len((set(col_x) - set(col_x).intersection(set(df.columns)))) != 0:
            raise ValueError("Please input the correct list of column names of x!")
        if len((set(confounder_cols) - set(confounder_cols).intersection(set(df.columns)))) != 0:
            raise ValueError("Please input the correct list of column names of confounders!")

    def linear_att(self):
        """
        Method for estimating the ATT with the ordinal least square linear model.
        Return the estimated ATT and print out the model summary.

        """

        X = sm.add_constant(self.df[[self.col_t] + self.col_x])
        y = np.asarray(self.df[self.col_y])
        self.wls = sm.WLS(y.astype(float), X.astype(float), weights=1).fit()
        print(self.wls.summary())

        return self.wls.params[self.col_t]

    def weighted_linear_att(self):
        """
        Method for estimating the ATT with the weighted least square linear model.
        Return the estimated ATT and print out the model summary.

        """

        X = sm.add_constant(self.df[[self.col_t] + self.col_x])
        y = np.asarray(self.df[self.col_y])
        try:
            weight = self.df['weight']
        except:
            print("There is no weight found in your dataframe! All samples will get weight=1 and the the result is equivalant with OLS linear model.")
            weight = np.repeat(1, len(X))

        self.wls = sm.WLS(y.astype(float), X.astype(float), weights = weight).fit()
        print(self.wls.summary())

        return self.wls.params[self.col_t]

    def linear_dml_hte(self, final_model: str = 'ols_linear'):
        """
        Method for estimating the HTE with the linear double machine learning method.
        Return the average treatment effect on treated ATT, conditional average treatment effect CATE, R2 score of the model.

        Parameter
        ---------
        final_model: str
            You can choose among ['ols_linear', 'lasso', 'ridge'] as your second stage model.

        """

        y = self.df[self.col_y]
        t = self.df[self.col_t]
        W = self.df[self.confounder_cols]
        self.col_hte_x = list(set(self.col_x) - set(self.confounder_cols).intersection(self.col_x))
        if len(self.col_hte_x) == 0:
            raise ValueError(
                "Please include column names of hetrogeneous X in 'col_x' if you want to get the hetrogeneous treatment effect!")
        X = self.df[self.col_hte_x]
        if len(self.col_hte_x) == 1:
            X = np.array(X).reshape(-1, 1)

        model_y = LR()
        model_y.fit(np.hstack([X, W]), y)
        y_res = y - model_y.predict(np.hstack([X, W]))
        y_res = np.array(y_res).reshape(-1, 1)

        model_t = LR()
        model_t.fit(np.hstack([X, W]), t)
        t_res = t - model_t.predict(np.hstack([X, W]))
        t_res = np.array(t_res).reshape(-1, 1)

        if final_model == 'ols_linear':
            model_result = LR(fit_intercept=False)
        elif final_model == 'lasso':
            model_result = LR(fit_intercept=False)
        elif final_model == 'ridge':
            model_result = LR(fit_intercept=False)
        else:
            raise ValueError(
                "Please choose a linear model among ['ols_linear', 'lasso', 'ridge'] for the second stage!")

        lm = model_result.fit(t_res, y_res)
        ate = lm.coef_
        loss_ate = np.mean((y_res - lm.predict(t_res)) ** 2)

        a = [np.multiply(X.iloc[:, i], t_res.flatten()) for i in range(len(self.col_hte_x))]
        a1 = [np.multiply(X.iloc[:, i], 1) for i in range(len(self.col_hte_x))]
        a0 = [np.multiply(X.iloc[:, i], 0) for i in range(len(self.col_hte_x))]
        dot_df = a[0]
        dot_df1 = a1[0]
        dot_df0 = a0[0]
        for i in range(1, len(a)):
            dot_df = np.vstack([dot_df, a[i]])
            dot_df1 = np.vstack([dot_df1, a1[i]])
            dot_df0 = np.vstack([dot_df0, a0[i]])
        dot_df = dot_df.T
        dot_df1 = dot_df1.T
        dot_df0 = dot_df0.T

        lm_cate = model_result.fit(np.hstack([t_res, dot_df]), y_res)
        cate = lm_cate.coef_[0][0]
        loss_cate = np.mean((y_res - lm_cate.predict(np.hstack([t_res, dot_df]))) ** 2)

        result1 = lm_cate.predict(np.hstack([np.repeat(1, len(t_res)).reshape(-1, 1), dot_df1]))
        result2 = lm_cate.predict(np.hstack([np.repeat(0, len(t_res)).reshape(-1, 1), dot_df0]))

        hte = result1 - result2

        r2 = 1 - loss_cate / loss_ate

        return cate, hte, r2