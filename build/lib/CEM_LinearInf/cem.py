import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from CEM_LinearInf.balance import balance


class cem:
    """
    Class of Coarsened Exact Matching (CEM).
    CEM is a data preprocessing algorithm in causal inference that has broad applicability to observational data.
    With CEM, you can construct your observational data into 'quasi' experimental data easily, mitigating the model dependency,
    bias, and inefficiency of your estimation of the treatment effect (King and Zeng 2006; Ho, Imai, King, and Stuart 2007; Iacus et al. 2008).

    Parameters
    ----------

    df: pd.Dataframe
        The dataframe you want to summarize.

    confounder_cols: list
        The column names of confounders among all variables X.

    cont_confounder_cols: list
        The column names of all continuous variables among confounders.

    col_y: string
        The column name of result Y in your dataframe. If not specified, it would be "Y".

    col_t: string
        The column name of treatment T in your dataframe. If not specified, it would be "T".

    Attributes
    ----------
    df: pd.Dataframe

    confounder_cols: list

    cont_confounder_cols: list

    dis_confounder_cols: list

    col_y: string

    col_t: string

    """

    def __init__(self,
                 df: pd.DataFrame,
                 confounder_cols: list,
                 cont_confounder_cols: list,
                 col_y: str = 'Y',
                 col_t: str = 'T',
                 ):

        self.df = df.copy()
        self.confounder_cols = confounder_cols
        self.cont_confounder_cols = cont_confounder_cols
        dis_confounder_cols = list(set(confounder_cols) - set(confounder_cols).intersection(cont_confounder_cols))
        self.dis_confounder_cols = dis_confounder_cols
        self.col_y = col_y
        self.col_t = col_t

    def summary(self):

        """
        Method for generating the summary of data, return the descriptive statistics result of the dataframe
        and the T-test result of Experimental group Y and Control group Y.

        """

        col_y = self.col_y
        col_t = self.col_t

        try:
            Y = self.df[col_y]
        except:
            raise ValueError(f"Cannot find column {col_y} in your dataframe!")

        try:
            T = self.df[col_t]
        except:
            raise ValueError(f"Cannot find column {col_t} in your dataframe!")

        print("Descriptive Statistics of the dataframe:\n")
        print(self.df.describe().applymap('{:,.4f}'.format))

        print("\nControl group vs. Experimental group \n")
        df_ce = pd.DataFrame(T.value_counts().values, columns=['n_samples'])
        df_ce['mean_Y'] = [self.df[self.df[col_t] == 0][col_y].mean(), self.df[self.df[col_t] == 1][col_y].mean()]
        print(df_ce)

        print("\nT-test of Experimental group Y and Control group Y\n")
        estimate = self.df[self.df[col_t] == 1][col_y].mean() - self.df[self.df[col_t] == 0][col_y].mean()
        _, pvalue = stats.ttest_ind(self.df[self.df[col_t] == 1][col_y], self.df[self.df[col_t] == 0][col_y])
        print(f"att estimate (p-value): {round(estimate, 4)}({round(pvalue, 4)})", )
        if pvalue >= 0.05:
            print("The difference between Experimental group Y and Control group Y is not significant.")
        elif 0 <= pvalue < 0.05:
            print(
                f"The difference between Experimental group Y and Control group Y is significant, and the difference is {round(estimate, 4)}.")

    def cut(self, col: pd.Series, func: str, param: int or list = None) -> pd.Series:

        """
        Method for cutting a continuous confounder X into discrete bins.

        Parameters
        ----------

        col: pd.Series
            Continuous confounder X to be coarsened.

        func: str
            Cutting function to be used. The following functions can be chosen.

            'cut': Bin values into discrete intervals with the same length.
            'qcut': Discretize variable into equal-sized buckets based on rank or based on sample quantiles.
            'struges': Bin values into discrete intervals with the same length k according to the Sturges' rule.

        param: int or list (optional)
            When the method is 'cut', it's a number of bins or a list of bins' edges;
            When the method is 'qcut', it's a number of quantiles or a list of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
            When the method is 'struges', there's no need to specify the param.

        Return
        ------
        Coarsened confounder X.

        """

        if func == 'cut':
            return pd.cut(x=col, bins=param, labels=False, retbins=False)

        elif func == 'qcut':
            return pd.qcut(x=col, q=param, labels=False, retbins=False)

        elif func == 'sturges':
            k = math.ceil(1 + np.log2(len(col)))
            return pd.cut(x=col, bins=k, labels=False, retbins=False)

        else:
            raise ValueError(
                f"'{func}' not supported. Coarsening only possible with 'cut', qcut', and 'struges'.")

    def coarsening(self, coarsen_df: pd.DataFrame = None, cont_confounder_cols: list = None,
                   schema: dict = {}) -> pd.DataFrame:

        """

        Coarsen your data based on the specified coarsen schema.
        If the coarsen schema is not specified, the default method is binning values into discrete intervals with the same length k according to the Sturges' rule.

        Parameters
        ----------

        coarsen_df: pd.DataFrame
            DataFrame to be coarsened.

        cont_confounder_cols: list
            Column names of continuous confounders of the dataframe to be coarsened.

        schema: dict
            The dictionary specifing what coarsening method you want to use on each continuous confounder X.
            If it is not specified, the default method is binning values into discrete intervals with the same length k according to the Sturges' rule.

        Return
        ------
        Dataframe with coarsened confounders X.

        """

        if schema is None:
            schema = {}
        if coarsen_df is None:
            coarsen_df = self.df
        if cont_confounder_cols is None:
            cont_confounder_cols = self.cont_confounder_cols

        temp_df = coarsen_df.copy()
        coarsen_columns = temp_df[cont_confounder_cols].apply(
            lambda x: self.cut(x, schema[x.name][0], schema[x.name][1]) if x.name in schema else self.cut(x, 'sturges'),
            axis=0)

        coarsen_cont_confounder_cols = ['coarsen_' + name for name in cont_confounder_cols]
        coarsen_df[coarsen_cont_confounder_cols] = coarsen_columns

        return coarsen_df

    def weight(self, strata_T: pd.Series, all_T: pd.Series) -> pd.Series:

        """

        Method of computing weights for each observation.

        Parameters
        ----------

        strata_T: pd.Series
            The treatment T of samples in one strata.

        all_T:  pd.Series
            The treatment T of all the matched samples.

        Return
        ------
        Weights of samples in the input strata.

        """

        all_counts = all_T.value_counts()
        strata_counts = strata_T.value_counts()
        T = all_T.max()

        strata_weights = [(all_counts[t] / strata_counts[t]) / (all_counts[T] / strata_counts[T]) for t in strata_T]
        strata_weights = pd.Series(strata_weights, index=strata_T.index)

        return strata_weights

    def match(self, schema: dict = {}, k2k_ratio: int = 0, dist: str = 'euclidean',
              _print: bool = True) -> pd.DataFrame:

        """

        Method for perform coarsened exact matching using the specified coarsening schema and return the dataframe with weights for each observation.
        If the coarsen schema is not specified, the default method is binning values into discrete intervals with the same length k according to the Sturges' rule.

        Parameters
        ----------

        schema: dict
            The dictionary specifing what coarsening method you want to use on each continuous confounder X.
            If it is not specified, the default method is binning values into discrete intervals with the same length k according to the Sturges' rule.

        k2k_ratio: int
            The ratio of # control samples / # treatment samples when we conduct k to k matching.
            If 'k2k_ratio' is 0, we don't conduct k to k matching.
            Otherwise, all treatment samples in each strata will be matched with k nearest control samples.

        dist: str
            The measure of distance when conducting k to k matching.
            You can choose among ['euclidean', 'mahalanobis;, 'psm']. The default one is 'euclidean'.

        _print: bool
            Whether to print the matching result.

        Return
        ------
        The matched dataframe.

        """

        confounder_cols = self.confounder_cols
        cont_confounder_cols = self.cont_confounder_cols
        dis_confounder_cols = self.dis_confounder_cols
        col_t = self.col_t
        col_y = self.col_y
        self.schema = schema
        # coarsening
        coarsen_df = self.coarsening(schema=self.schema)

        group = dis_confounder_cols + ['coarsen_' + name for name in cont_confounder_cols]
        matched = coarsen_df.groupby(group).filter(lambda x: x[col_t].nunique() == self.df[col_t].nunique())

        if not len(matched):
            raise ValueError('There is no matched pair! Please reduce confounder variables or try coarser bins.')

        weights = pd.concat([self.weight(strata[col_t], matched[col_t]) for _, strata in matched.groupby(group)])
        self.df['weight'] = weights
        self.matched_df = self.df[pd.notna(self.df['weight'])]

        if k2k_ratio != 0:
            if dist not in ['euclidean', 'mahalanobis', 'psm']:
                raise ValueError("Only 'euclidean', 'mahalanobis' and 'psm' distances are supported!")

            k2k_matched_df, k2k_pair = self.k2k_match(self.matched_df, self.confounder_cols, group, dist, k2k_ratio)
            self.matched_df = k2k_matched_df
            self.pair = k2k_pair
            self.df.drop('weight', axis=1, inplace=True)
            self.matched_df.drop('weight', axis=1, inplace=True)

        if _print:
            all_counts = pd.DataFrame(self.df[self.col_t].value_counts().values, columns=['all'])
            matched_counts = pd.DataFrame(self.matched_df[self.col_t].value_counts().values, columns=['matched'])
            count_df = all_counts.merge(matched_counts, left_index=True, right_index=True)
            count_df['propotion'] = round(count_df['matched'] / count_df['all'], 4)

            print('Matching result\n')
            print(f'{count_df}\n')

        return self.matched_df

    def k2k_match(self, df: pd.DataFrame, confounder_cols: list, group: list, dist: str, k2k_ratio: int = 1):
        """

        Method for performing k to k coarsened exact matching using the specified distance measure,
        and return the dataframe with weights for each observation.

        Parameters
        ----------

        df: pd.DataFrame
            The coarsened dataframe to be matched.

        confounder_cols: list
            The column names of all continuous variables among confounders.

        group: list
            The column names of variables used for grouping.
            It's equivalent to dis_confounder_cols + ['coarsen_' + name for name in cont_confounder_cols].

        dist: str
            The measure of distance when conducting k to k matching.
            You can choose among ['euclidean', 'mahalanobis;, 'psm']. The default one is 'euclidean'.

        k2k_ratio: int
            The ratio of # control samples / # treatment samples when we conduct k to k matching.
            All treatment samples in each strata will be matched with k nearest control samples.
            The default ratio is 1.

        Return
        ------
        The matched dataframe.
        A dictionary of pair index, indicating which control sample is paired with each experimantal sample.

        """

        matched_df = pd.DataFrame([], columns=self.matched_df.columns)
        pair = {}

        for _, sub_df in df.groupby(group):

            if dist == 'psm':
                X, y = sub_df[self.confounder_cols], sub_df[self.col_t]
                clf = LogisticRegression(random_state=0).fit(X, y)
                sub_df['score'] = clf.predict_proba(X)[:, 1]

            df_tr = sub_df[sub_df[self.col_t] == 1]
            df_ct = sub_df[sub_df[self.col_t] == 0]

            if dist in ['euclidean', 'mahalanobis']:
                neigh = NearestNeighbors(n_neighbors=k2k_ratio, n_jobs=-1, metric=dist)
                neigh.fit(df_ct[confounder_cols])
                distances, indices = neigh.kneighbors(df_tr[confounder_cols])

            elif dist == 'psm':
                neigh = NearestNeighbors(n_neighbors=k2k_ratio, n_jobs=-1)
                neigh.fit(np.array(df_ct['score']).reshape(-1, 1))
                distances, indices = neigh.kneighbors(np.array(df_tr['score']).reshape(-1, 1))

            matched_df = pd.concat([matched_df, df_tr])
            matched_df = pd.concat([matched_df, df_ct.iloc[indices.flatten(), :]])

            for i in range(len(indices)):
                pair[df_tr.index[i]] = df_ct.iloc[indices[i], :].index[0]

        return matched_df, pair

    def tunning_schema(self, step: int = 4):

        """

        Method for fine-tunning continuous confounders' binning schema automatically. The optimization objective is to have a smaller L1 score
        conditional on a relatively large matched sample size.

        Parameters
        ----------

        step: int
            The step when tunning the number of bins.
            E.X. When 'step' = 4, the function will calculate all L1 scores when the number of bins equals [4, 8, ..., 1 + log2(n)].

        Return
        ------
        A dataframe recording L1 scores of each schemas.
        The optimal schema dictionary.

        """

        X, y = self.df[self.cont_confounder_cols], self.df[self.col_t]
        clf = LogisticRegression(random_state=0).fit(X, y)
        order = np.argsort(-abs(clf.coef_))

        method = ['cut', 'qcut']
        k = math.ceil(1 + np.log2(len(self.df)))
        length = range(step, k, step)
        l1_df = pd.DataFrame([], columns=['confounder', 'method', 'step', 'matched_num', 'l1'])
        schema = {}

        self.match(_print = False)
        my_balance = balance(self.matched_df, self.df, self.confounder_cols, self.cont_confounder_cols, self.col_y, self.col_t)
        _, l1_bench = my_balance.balance_assessing(method='L1', _print=False)
        num_bench = len(self.matched_df) - len(self.df) / 20
        l1_min = l1_bench

        for i in range(len(order)):
            x = self.cont_confounder_cols[order[0][i]]
            l1_list = []
            schema_temp = {}

            if len(l1_df) > 0:
                for j in l1_df.index:
                    schema_temp[l1_df.loc[j, 'confounder']] = (l1_df.loc[j, 'method'], l1_df.loc[j, 'step'])

            for m in method:
                for s in length:
                    temp_df = self.df.copy()
                    my_cem1 = cem(temp_df, self.confounder_cols, self.cont_confounder_cols, self.col_y, self.col_t)
                    schema_temp_new = {x: (m ,s)}
                    schema_temp_new.update(schema_temp)
                    match_df1 = my_cem1.match(schema = schema_temp_new, _print = False)
                    matched_num = len(match_df1)
                    my_balance1 = balance(match_df1, my_cem1.df, my_cem1.confounder_cols, my_cem1.cont_confounder_cols, self.col_y, self.col_t)
                    _, l1_new = my_balance1.balance_assessing(method = 'L1', _print = False)
                    if (l1_new < l1_bench) and (matched_num >= num_bench):
                        l1_list.append([x, m, s, matched_num, l1_new])

            if (len(l1_list) > 0):

                l1_df_temp = pd.DataFrame(l1_list, columns=['confounder', 'method', 'step', 'matched_num', 'l1'])
                l1_min_temp = l1_df_temp['l1'].min()
                l1_df_temp = l1_df_temp[l1_df_temp['l1'] == l1_min_temp]

                if l1_min_temp < l1_min:
                    l1_min = l1_min_temp
                    l1_df = pd.concat([l1_df, l1_df_temp])
            else:
                pass

        for j in l1_df.index:
            schema[l1_df.loc[j, 'confounder']] = [l1_df.loc[j, 'method'], l1_df.loc[j, 'step']]

        return l1_df, schema
