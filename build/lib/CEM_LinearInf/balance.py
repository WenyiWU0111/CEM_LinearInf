import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class balance:

    """
    Class of balance assessment for the matched data.

    When we finish the coarsened exact matching, it is necessary to evaluate the quality of the matching
    with imbalance checking methods. When the covariate balance is achieved, the resulting effect estimate
    is less sensitive to model misspecification and ideally close to true treatment effect (Greifer, 2023).

    The imbalance checking methods provided include:
        'L1': Calculate and return the L1 imbaance score.
        'smd': Print the standardized mean difference summary table and plots of confounders.
        'ks': Plot Kolmogorov-Smirnov Statistics of confounders before and after matching.
        'density': Return density plots of confounders before and after matching.
        'ecdf': Return empirical cumulative density plots of confounders before and after matching.

    Parameters
    ----------

    df_match: pd.Dataframe
        The dataframe after matching.

    df_all: pd.Dataframe
        The original dataframe before matching.

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
    df_match: pd.Dataframe

    df_all: pd.Dataframe

    confounder_cols: list

    cont_confounder_cols: list

    dis_confounder_cols: list

    col_y: string

    col_t: string

    """

    def __init__(self,
                 df_match: pd.DataFrame,
                 df_all: pd.DataFrame,
                 confounder_cols: list,
                 cont_confounder_cols: list,
                 col_y: str = 'Y',
                 col_t: str = 'T'):

        self.df_match = df_match.copy()
        self.df_all = df_all.copy()
        self.confounder_cols = confounder_cols
        self.cont_confounder_cols = cont_confounder_cols
        disc_confounder_cols = list(set(confounder_cols) - set(confounder_cols).intersection(cont_confounder_cols))
        self.disc_confounder_cols = disc_confounder_cols
        self.col_y = col_y
        self.col_t = col_t

        self.df_all.fillna(0, inplace=True)

    def compute_l1(self, df_tr: pd.DataFrame, df_ct: pd.DataFrame, cont_confounder_cols: list,
                   disc_confounder_cols: list) -> float:

        group = disc_confounder_cols + ['coarsen_' + name for name in cont_confounder_cols]
        prop_tr = np.array([[len(strata) / len(df_tr), _] for _, strata in df_tr.groupby(group)], dtype=object)
        prop_ct = np.array([[len(strata) / len(df_ct), _] for _, strata in df_ct.groupby(group)], dtype=object)

        tr_df = pd.DataFrame(prop_tr[:, 0], index=prop_tr[:, 1], columns=['tr'])
        ct_df = pd.DataFrame(prop_ct[:, 0], index=prop_ct[:, 1], columns=['ct'])
        merge_df = tr_df.merge(ct_df, left_index=True, right_index=True, how='outer')
        merge_df.fillna(0, inplace=True)

        l1 = sum(abs(merge_df['tr'] - merge_df['ct'])) / 2

        return l1

    def compute_smd(self, df_tr: pd.DataFrame, df_ct: pd.DataFrame, confounder_cols: list):

        mean_tr = df_tr[confounder_cols].apply(np.mean)
        mean_ct = df_ct[self.confounder_cols].apply(lambda x: np.average(x, weights=df_ct['weight']))
        mean_diff = mean_tr - mean_ct

        var_tr = df_tr[confounder_cols].apply(np.var)
        var_ct = df_ct[self.confounder_cols].apply(
            lambda x: np.average((x - np.average(x, weights=df_ct['weight'])) ** 2, weights=df_ct['weight']))
        pooled_var = (var_tr * len(df_tr) + var_ct * len(df_ct)) / (len(df_tr) + len(df_ct) - 2)

        smd = mean_diff / np.sqrt(pooled_var)
        var_ratio = (var_tr + 1e-6) / (var_ct + 1e-6)

        return mean_tr, mean_ct, mean_diff, var_tr, var_ct, pooled_var, smd, var_ratio

    def balance_assessing(self, method: str = 'smd', threshold: list = [0.1, 2], _print: bool = True):

        """

        Method for generating the imbalance assessing result.

        Parameters
        ----------

        method: str
            The method to be used for balance assessment. If it's not specified, the default method is 'smd'.

            'L1': Calculate and return the L1 imbaance score.
            'smd': Print the standardized mean difference summary table and plots of confounders.
            'ks': Plot Kolmogorov-Smirnov Statistics of confounders before and after matching.
            'density': Return density plots of confounders before and after matching.
            'ecdf': Return empirical cumulative density plots of confounders before and after matching.
            'all': Implement all the methods above.

        threshold: list
            When you choose 'smd' to assess the balance, you can set the balance thresholds for smd and variance ratio.
            If it's not specified, the default thresholds are 0.1 and 2 for standardized mean difference and variance ratio respectively.

        _print: bool
            Whether to print the L1 score.

        """

        df_match = self.df_match.copy()
        df_all = self.df_all.copy()

        if 'weight' not in df_match.columns:
            df_match['weight'] = 1
            df_all['weight'] = 1

        df_tr = df_match[df_match[self.col_t] == 1]
        df_ct = df_match[df_match[self.col_t] == 0]
        df_tr_all = df_all[df_all[self.col_t] == 1]
        df_ct_all = df_all[df_all[self.col_t] == 0]

        if method == 'L1':

            l1_all = self.compute_l1(df_tr_all, df_ct_all, self.cont_confounder_cols, self.disc_confounder_cols)
            l1_match = self.compute_l1(df_tr, df_ct, self.cont_confounder_cols, self.disc_confounder_cols)

            if _print:
                print(f"L1 imbalance score before matching: {round(l1_all, 4)}\n")
                print(f"L1 imbalance score after matching: {round(l1_match, 4)}\n")

            return l1_all, l1_match

        elif method == 'smd':

            mean_tr, mean_ct, mean_diff, var_tr, var_ct, pooled_var, smd, var_ratio = self.compute_smd(df_tr, df_ct,
                                                                                                       self.confounder_cols)
            mean_tr_all, mean_ct_all, mean_diff_all, var_tr_all, var_ct_all, pooled_var_all, smd_all, var_ratio_all = self.compute_smd(
                df_tr_all, df_ct_all, self.confounder_cols)

            imbalance_df = pd.DataFrame({
                'Treated Mean': round(mean_tr, 4),
                'Control Mean': round(mean_ct, 4),
                'SMD': round(smd, 4),
                'Variance Ratio': round(var_ratio, 4)}, index=self.confounder_cols)

            mean_thresh = threshold[0]
            var_thresh = threshold[1]
            mean_thresh_str = 'SMD.Threshold(<' + str(mean_thresh) + ')'
            var_thresh_str = 'Var.Threshold(<' + str(var_thresh) + ')'

            imbalance_df[[mean_thresh_str]] = 'Balanced'
            imbalance_df.loc[imbalance_df['SMD'] >= mean_thresh, mean_thresh_str] = 'Not balanced'

            imbalance_df[[var_thresh_str]] = 'Balanced'
            imbalance_df.loc[(imbalance_df['Variance Ratio'] >= var_thresh), var_thresh_str] = 'Not balanced'
            imbalance_df.loc[self.disc_confounder_cols, ['Variance Ratio', var_thresh_str]] = '.'

            print('Balance measures\n')
            print(imbalance_df)

            print('\n-------------------------')
            print('Balance tally for SMD\n')
            print(pd.DataFrame(imbalance_df[[mean_thresh_str]].value_counts(),
                               columns=['count']))

            print('\n------------------------------')
            print('Variable with the max SMD:\n')
            print(imbalance_df.loc[imbalance_df['SMD'] == max(imbalance_df['SMD']), ['SMD', mean_thresh_str]])

            print('\n------------------------------------')
            print('Balance tally for Variance ratio\n')
            print(pd.DataFrame(imbalance_df.loc[self.cont_confounder_cols, [var_thresh_str]].value_counts(),
                               columns=['count']))

            print('\n-----------------------------------------')
            print('Variable with the max variance ratio:\n')
            print(imbalance_df.loc[imbalance_df['Variance Ratio'] == max(
                imbalance_df.loc[self.cont_confounder_cols, 'Variance Ratio']), ['Variance Ratio', var_thresh_str]])

            print('\n-----------------------------------------\n')
            plot_df = pd.DataFrame({"Matched samples": smd,
                                    "All samples": smd_all})
            sns.set(font_scale=1.1)
            plot = sns.scatterplot(data=plot_df, s=50)
            plot.set(title='SMD Plot', xlabel='Confounders', ylabel='SMD')
            plt.axhline(y=0, color='grey', linestyle='--')
            plt.setp(plot.get_legend().get_texts(), fontsize='10')
            plt.show()

            return imbalance_df

        elif method == 'ks':

            all_ks = []
            matched_ks = []

            for i in range(len(self.confounder_cols)):
                col = self.confounder_cols[i]
                matched_ks.append(stats.ks_2samp(self.df_match.loc[self.df_match[self.col_t] == 0, col],
                                                 self.df_match.loc[self.df_match[self.col_t] == 1, col])[0])
                all_ks.append(stats.ks_2samp(self.df_all.loc[self.df_all[self.col_t] == 0, col],
                                             self.df_all.loc[self.df_all[self.col_t] == 1, col])[0])

            plot_df = pd.DataFrame({"Matched samples": matched_ks,
                                    "All samples": all_ks})

            plt.figure(figsize=(8, 5))
            sns.set(font_scale=1.1)
            plot = sns.scatterplot(data=plot_df, s=50)
            plot.set(title='Kolmogorov-Smirnov Statistic Plot', xlabel='Confounders', ylabel='KS')
            plt.axhline(y=0, color='grey', linestyle='--')
            plt.legend(loc=1)
            plt.setp(plot.get_legend().get_texts(), fontsize='10')
            plt.show()

        elif method == 'density':

            len_cont = len(self.cont_confounder_cols)
            len_disc = len(self.disc_confounder_cols)

            plt.figure(figsize=(10, 3 * len_cont))

            for i in range(len_cont):
                x = self.cont_confounder_cols[i]

                plot1 = plt.subplot(len_cont, 2, (2 * i + 1))
                sns.kdeplot(data=df_ct_all, x=x, fill=True)
                sns.kdeplot(data=df_tr_all, x=x, fill=True)

                plot2 = plt.subplot(len_cont, 2, (2 * i + 2))
                sns.kdeplot(data=df_ct, x=x, fill=True)
                sns.kdeplot(data=df_tr, x=x, fill=True)

                if i == 0:
                    plot1.set(title='All', xlabel='', ylabel=x)
                    plot2.set(title='Matched', xlabel='', ylabel='')
                else:
                    plot1.set(xlabel='', ylabel=x)
                    plot2.set(xlabel='', ylabel='')

            plt.figure(figsize=(10, 3 * len_disc))

            with sns.color_palette("ch:s=.25,rot=-.25"):

                for i in range(len_disc):
                    col = self.disc_confounder_cols[i]
                    types = len(self.df_all[[col]].value_counts())

                    plot1 = plt.subplot(len_disc, 2, (2 * i + 1))
                    sns.barplot(df_ct_all[[col]].apply(lambda x: x.value_counts() / len(x), axis=0).T,
                                order=range(types), alpha=0.7, linewidth=1, edgecolor=".5")
                    sns.barplot(df_tr_all[[col]].apply(lambda x: x.value_counts() / len(x), axis=0).T,
                                order=range(types), alpha=0.7, linewidth=1, edgecolor=".5")

                    plot2 = plt.subplot(len_disc, 2, (2 * i + 2))
                    sns.barplot(df_ct[[col]].apply(lambda x: x.value_counts() / len(x), axis=0).T, order=range(types),
                                alpha=0.7, linewidth=1, edgecolor=".5")
                    sns.barplot(df_tr[[col]].apply(lambda x: x.value_counts() / len(x), axis=0).T, order=range(types),
                                alpha=0.7, linewidth=1, edgecolor=".5")

                    plot1.set(xlabel='', ylabel=col)
                    plot2.set(xlabel='', ylabel='')

            plt.show()

        elif method == 'ecdf':

            len_con = len(self.confounder_cols)

            plt.figure(figsize=(10, 3 * len_con))

            for i in range(len_con):
                x = self.confounder_cols[i]

                plot1 = plt.subplot(len_con, 2, (2 * i + 1))
                sns.ecdfplot(data=df_ct_all, x=x)
                sns.ecdfplot(data=df_tr_all, x=x)

                plot2 = plt.subplot(len_con, 2, (2 * i + 2))
                sns.ecdfplot(data=df_ct, x=x)
                sns.ecdfplot(data=df_tr, x=x)

                if i == 0:
                    plot1.set(title='All', xlabel='', ylabel=x)
                    plot2.set(title='Matched', xlabel='', ylabel='')
                else:
                    plot1.set(xlabel='', ylabel=x)
                    plot2.set(xlabel='', ylabel='')

            plt.show()

        elif method == 'all':

            self.balance_assessing(method='L1')

            print("-------------------------\n")
            print("SMD Result\n")
            self.balance_assessing(method='smd', threshold=threshold)

            print("-------------------------\n")
            print("KS Result\n")
            self.balance_assessing(method='ks')

            print("-------------------------\n")
            print("Density Plot\n")
            self.balance_assessing(method='density')

            print("-------------------------\n")
            print("ECDF Plot\n")
            self.balance_assessing(method='ecdf')

        else:
            raise ValueError(
                "Please choose the balance assessing method among 'L1', 'smd', 'ks', 'density', 'ecdf', and 'all'!")
