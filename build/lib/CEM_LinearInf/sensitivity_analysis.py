import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels.api as sm


class wilcoxon:

    """

    Class of wilcoxon's signed rank test based sensitivity analysis methods.

    When we conduct causal inference to the observational data, the most important assumption is that there
    is no unobserved confounding.
    Therefore, after finishing the treatment effect estimation, investigators are advised to conduct the sensitivity analysis
    to examine how fragile a result is against the possibility of unobserved confounders (Cinelli, Hazlett, 2019).
    In other words, we should examine how strong the effect of unobserved confounders should be to erase the treatment effect estimated.

    Wilcoxon's signed rank test based sensitivity analysis method is provided in this class.
    Please be noted that you can only use this method if your data is 1-1 matched.

    Parameters
    ----------

    df: pd.DataFrame
        The matched dataframe.

    pair: dict
        A dictionary of pair index, indicating which control sample is paired with each experimantal sample.

    col_y: str
        The column name of dependent variable Y in your dataframe. If not specified, it would be "Y".

    """

    def __init__(self, df: pd.DataFrame, pair: dict, col_y: str = 'Y'):

        self.df = df.copy()
        self.pair = pair
        self.col_y = col_y

    def result(self, gamma_list: list):

        """
        Method to conduct the wilcoxon's signed rank test based sensitivity analysis.

        Parameters
        ----------
       gamma_list: list
            A list of gamma values you want to test. Gamma refers to the possibility a sample will be treated
            compared to its pair sample.

        Return
        ------
        A result dataframe including p-values under each gamma value.

        """

        result_df = pd.DataFrame([], columns=['gamma', 'lower_p', 'upper_p'])
        diff_list = []
        for key in self.pair:
            diff = self.df.loc[key, self.col_y] - np.max(self.df.loc[self.pair[key], self.col_y])
            diff_list.append(diff)

        s = len(diff_list)
        orders = pd.Series(diff_list).rank()
        orders_df = pd.DataFrame({'diff': diff_list, 'order': orders})
        w = orders_df[orders_df['diff'] > 0]['order'].sum()
        #print(w)

        #miu = np.sum(diff_list)

        for gamma in gamma_list:
            lam = gamma / (1 + gamma)
            miu_max = lam * s * (s + 1) / 2
            miu_min = (1 - lam) * s * (s + 1) / 2
            variance = lam * (1 - lam) * s * (s + 1) * (2 * s + 1) / 6
            #print(miu_max)

            z_upper = (w - miu_max) / np.sqrt(variance)
            z_lower = (w - miu_min) / np.sqrt(variance)

            p_upper = norm.sf(z_upper)
            #print(p_upper)
            p_lower = norm.sf(z_lower)
            # prob_up = gamma / (1 + gamma)
            # prob_low = 1 / (1 + gamma)
            # upper_list = []
            # lower_list = []
            #
            # for i in range(1000):
            #     factor_upper = 2 * bernoulli.rvs(prob_up, size=len(diff_list)) - 1
            #     factor_lower = 2 * bernoulli.rvs(prob_low, size=len(diff_list)) - 1
            #
            #     upper_list.append(np.sum(factor_upper * diff_list))
            #     lower_list.append(np.sum(factor_lower * diff_list))

            # lower_mean = np.mean(lower_list)
            # lower_std = np.std(lower_list)
            # upper_mean = np.mean(upper_list)
            # upper_std = np.std(upper_list)
            #
            # stat_upper = (upper_mean - miu) / upper_std
            # stat_lower = (lower_mean - miu) / lower_std
            #
            # p_upper = 2 * t.sf(abs(stat_upper), len(upper_list) - 1)
            # p_lower = 2 * t.sf(abs(stat_lower), len(lower_list) - 1)

            new_result = pd.DataFrame({'gamma': [gamma],
                                          'lower_p': [round(p_lower, 4)],
                                          'upper_p': [round(p_upper, 4)]})
            result_df = pd.concat([result_df, new_result])

        reset_index = result_df.set_index('gamma', inplace=True)
        print(result_df)

        try:
            thresh_gamma = result_df[result_df['upper_p'] >= 0.05].iloc[0, :].name
            print(
                'The estimated ATT result is not reliable if there exists an unobservable confounder which makes the magnitude of probability')
            print(
                f'that a single subject will be interfered with is {thresh_gamma} times higher than that of the other subject.')

        except:
            print("All gamma values pass the wilcoxon's sensitivity analysis, and the estimated ATT result is reliable.")

        return result_df


class ovb:
    """
    Class of omitted variable bias based sensitivity analysis methods.

    When we conduct causal inference to the observational data, the most important assumption is that there
    is no unobserved confounding.
    Therefore, after finishing the treatment effect estimation, investigators are advised to conduct the sensitivity analysis
    to examine how fragile a result is against the possibility of unobserved confounders (Cinelli, Hazlett, 2019).
    In other words, we should examine how strong the effect of unobserved confounders should be to erase the treatment effect estimated.

    Omitted variable bias based sensitivity analysis method is provided in this class.
    Please be noted that you can only use this method in linear case.

    Parameters
    ----------
    col_t: str
        The column name of treatment variable T in your dataframe. If not specified, it would be "T".

    model:
        The regression model before fitted.

    bench_variable: str
        The confounder you choose as a benchmark.

    k_t: int or list
        R2 between treatment and the unobservable confounder / R2 between treatment and the benchmark confounder
        You can interpret as how many times the correlation between the unobservable confounder and the treatment is to
        that between the benchmark confounder and the treatment. The default value is 1.

    k_y: int or list
        R2 between result y and the unobservable confounder / R2 between result y and the benchmark confounder
        You can interpret as how many times the correlation between the unobservable confounder and the result is to
        that between the benchmark confounder and the result. The default value is 1.

    threshold: int
        The threshold level in the result plot. The default value is 0.

    measure: str
        The measure you want to shown in the result plot. You can choose between 'att' and 't'. The default measure is 'att'.
        'att': The estimated average treatment effect.
        't': The t-value of the estimated average treatment effect.

    """

    def __init__(self, col_t: str ='T', model = None,
                 bench_variable: str = None, k_t: int or list = 1, k_y: int or list = 1,
                 measure: str = 'att'):

        try:
            self.model_data = pd.DataFrame(model.exog, columns=model.exog_names)
        except:
            raise ValueError("Please input the regression model before fitting!")

        self.wls = model.fit()
        self.col_t = col_t
        self.k_t = k_t
        self.k_y = k_y
        self.bench_variable = bench_variable
        self.measure = measure

    def calculate_ovb(self, r1_squred: float, r2_squred: float):
        """
        Method for calculating the omitted variable bias.

        """
        ovb = np.sqrt(r1_squred * r2_squred / (1 - r1_squred)) * self.se * np.sqrt(self.df_resid)
        return ovb

    def adj_estimate(self, r1_squred, r2_squred):
        """
        Method for calculating the adjusted ATT estimation.

        """
        ovb = self.calculate_ovb(r1_squred, r2_squred)
        return np.sign(self.estimate) * (abs(self.estimate) - ovb)

    def adj_t(self, r1_squred, r2_squred):
        """
        Method for calculating the adjusted t-value of ATT estimation.

        """
        adj_est = self.adj_estimate(r1_squred, r2_squred)
        return (adj_est - self.miu) / self.se

    def calculate_rv(self):

        f2_yd_x = self.calculate_f2(self.t)
        RV = (np.sqrt(f2_yd_x ** 2 + 4 * f2_yd_x) - f2_yd_x) / 2
        return RV

    def calculate_f2(self, t):
        return t ** 2 / self.wls.df_resid

    def calculate_r2(self, t):
        return t ** 2 / (t ** 2 + self.wls.df_resid)

    def calculate_bound(self, k_t: int, k_y: int, r2_dxj_x: float, f2_dxj_x: float):

        r2_dz_x = k_t * f2_dxj_x
        r2_zxj_dx = k_t * r2_dxj_x ** 2 * (1 - r2_dxj_x) / (1 - k_t * r2_dxj_x)
        if r2_zxj_dx >= 1:
            raise ValueError("R2 is inflated! Please try other confounders or smaller k_t, k_y.")

        f2_yxj_dx = self.calculate_f2(self.wls.tvalues[self.bench_variable])
        r2_yz_dx = (np.sqrt(k_y) + np.sqrt(r2_zxj_dx)) ** 2 / (1 - r2_zxj_dx) * f2_yxj_dx
        if r2_yz_dx >= 1:
            print(f"R2 = {r2_yz_dx}")
            raise ValueError("R2 is inflated! Please try other confounders or smaller k_t, k_y.")

        if self.measure == 'att':
            bound = self.adj_estimate(r2_yz_dx, r2_dz_x)
        elif self.measure == 't':
            bound = self.adj_t(r2_yz_dx, r2_dz_x)
        else:
            raise ValueError("The measure you input is not valid1 Please choose one from ['att', 't']")

        return r2_yz_dx, r2_dz_x, bound

    def get_bound(self):

        if isinstance(self.k_t, int):
            k_t = [self.k_t]
        else:
            k_t = self.k_t

        if isinstance(self.k_y, int):
            k_y = [self.k_y]
        else:
            k_y = self.k_y

        treat = np.asarray(self.model_data[self.col_t])
        X_treat = self.model_data.drop(columns=self.col_t)
        wls_treat = sm.WLS(treat.astype(float), X_treat.astype(float), weights=1).fit()

        r2_dxj_x = self.calculate_r2(wls_treat.tvalues[self.bench_variable])
        f2_dxj_x = self.calculate_f2(wls_treat.tvalues[self.bench_variable])

        r2_yz_dx_list = []
        r2_dz_x_list = []
        bound_list = []
        for kd in k_t:
            for ky in k_y:
                r2_yz_dx, r2_dz_x, bound = self.calculate_bound(kd, ky, r2_dxj_x, f2_dxj_x)
                bound_list.append(round(bound, 4))
                r2_yz_dx_list.append(round(r2_yz_dx, 4))
                r2_dz_x_list.append(round(r2_dz_x, 4))

        return r2_yz_dx_list, r2_dz_x_list, bound_list

    def plot_result(self, lim_x: float = 0.8, lim_y: float = 0.8, threshold: int or float = 0):

        """
        Method for omiited variable bias based sensitivity analysis.
        The result plot presents how the averaged treatment effect or the t-value will change with
        different values of R2 between unobservable confounder and T and that between unobservable confounder and Y

        Parameters
        ----------
        lim_x: float
            x-axis limit of the plot.

        lim_y: float
            y-axis limit of the plot.

        """
        self.se = self.wls.bse[self.col_t]
        self.df_resid = self.wls.df_resid
        self.estimate = self.wls.params[self.col_t]
        self.t = self.wls.tvalues[self.col_t]
        self.miu = self.estimate - self.se * self.t

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        plt.xlim(-0.05, lim_x)
        plt.ylim(-0.05, lim_y)
        grid_values_x = np.arange(0, lim_x, lim_x / 400)
        grid_values_y = np.arange(0, lim_y, lim_y / 400)

        if self.measure == 'att':
            ori_value = self.estimate
            z_axis = [[self.adj_estimate(grid_values_x[j], grid_values_y[i])
                       for j in range(len(grid_values_x))] for i in range(len(grid_values_y))]
        elif self.measure == 't':
            ori_value = self.t
            z_axis = [[self.adj_t(grid_values_x[j], grid_values_y[i])
                       for j in range(len(grid_values_x))] for i in range(len(grid_values_y))]
        else:
            raise ValueError("The measure you input is not valid1 Please choose one from ['att', 't']")

        CS1 = ax.contour(grid_values_x, grid_values_y, z_axis,
                         colors='dimgrey', linewidths=1.0, linestyles="solid", levels=10)
        plt.scatter(0, 0, marker='*', s=2 ** 7, alpha=0.7)

        plt.text(0.02, -0.02, f'Original {self.measure} \n({round(ori_value, 4)})', size=12, fontname='serif')
        ax.clabel(CS1, inline=True, fontsize=12)
        plt.xlabel('Partial R2 of unobservable confounder with the treatment T', size=12, fontname='serif')
        plt.ylabel('Partial R2 of unobservable confounder with the outcome Y', size=12, fontname='serif')

        if threshold in CS1.levels:
            CS2 = ax.contour(grid_values_x, grid_values_y, z_axis,
                             colors='red', linewidths=1.0, linestyles="dashed", levels=[threshold])
            ax.clabel(CS2, inline=True, fontsize=12)
        else:
            raise ValueError("No contour found in the given R2 scope, please try smaller threshold or larger lim_x, lim_y.")

        rv = self.calculate_rv()
        plt.scatter(rv, rv, marker='*', s=2 ** 7, color='red', alpha=0.7)
        plt.text(rv - 0.15, rv - 0.02, f'RV: \n({round(rv, 4)})', size=12, fontname='serif')

        if self.bench_variable is not None:
            if self.bench_variable in self.model_data.columns:
                r2_1_list, r2_2_list, bound_list = self.get_bound()

                plt.scatter(r2_2_list[0], r2_1_list[0], marker='^', s=2 ** 7, color='red', alpha=0.7)
                plt.text(r2_2_list[0] - 0.02, r2_1_list[0] + 0.05,
                         f'({self.k_t[0]}, {self.k_y[0]}) x {self.bench_variable} \n({round(bound_list[0], 4)})', size=12,
                         fontname='serif')

                bound_df = pd.DataFrame([], columns=['K_t', 'K_y'])
                for kd in self.k_t:
                    for ky in self.k_y:
                        new_bound = pd.DataFrame({'K_t': [kd], 'K_y': [ky]})
                        bound_df = pd.concat([bound_df, new_bound])
                        #bound_df = bound_df.append({'K_t': kd, 'K_y': ky}, ignore_index=True)

                bound_df['R2_Y'] = r2_1_list
                bound_df['R2_T'] = r2_2_list
                bound_df[f'Adjusted {self.measure}'] = bound_list

                return bound_df

            else:
                raise ValueError("Please input a valid name of the benchmark variable among confounders!")

        plt.tight_layout()
