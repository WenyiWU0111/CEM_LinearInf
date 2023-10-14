Sensitivity Analysis 
====================

Wilcoxon's signed rank test based sensitivity analysis
---------------------------------------------------------------

.. py:class:: wilcoxon(df: pd.DataFrame, pair: dict, col_y: str = 'Y')

   Class of wilcoxon's signed rank test based sensitivity analysis methods.

   When we conduct causal inference to the observational data, the most important assumption is that there is no unobserved confounding.
   
   Therefore, after finishing the treatment effect estimation, investigators are advised to conduct the sensitivity analysis to examine how fragile a result is against the possibility of unobserved confounders (Cinelli, Hazlett, 2019).
   
   In other words, we should examine how strong the effect of unobserved confounders should be to erase the treatment effect estimated.

   Wilcoxon's signed rank test based sensitivity analysis method is provided in this class. Please be noted that you can only use this method if your data is 1-1 matched.

   :param pd.DataFrame df: 
        The matched dataframe.

   :param dict pair: 
        A dictionary of pair index, indicating which control sample is paired with each experimantal sample.

   :param str col_y: 
        The column name of dependent variable Y in your dataframe. If not specified, it would be "Y".

   .. method:: result(self, gamma_list: list)

      Method to conduct the wilcoxon's signed rank test based sensitivity analysis.

      :param gamma_list: list
         A list of gamma values you want to test. Gamma refers to the possibility a sample will be treated compared to its pair sample.

      :returns:
         pd.DataFrame: A result dataframe including p-values under each gamma value.


Omitted variable bias based sensitivity analysis
-------------------------------------------------------------------

.. py:class:: ovb()

   Class of omitted variable bias based sensitivity analysis methods.

   When we conduct causal inference to the observational data, the most important assumption is that there is no unobserved confounder.
   
   Therefore, after finishing the treatment effect estimation, investigators are advised to conduct the sensitivity analysis to examine how fragile a result is against the possibility of unobserved confounders (Cinelli, Hazlett, 2019).
   
   In other words, we should examine how strong the effect of unobserved confounders should be to erase the treatment effect estimated.

   Omitted variable bias based sensitivity analysis method is provided in this class. Please be noted that you can only use this method in linear case.

   :param str col_t: 
        The column name of treatment variable T in your dataframe. If not specified, it would be "T".

   :param model:
        The regression model before fitted.

   :param str bench_variable: 
        The confounder you choose as a benchmark.

   :param int or list k_t:
     
     :math:`R^2` between treatment and the unobservable confounder / :math:`R^2` between treatment and the benchmark confounder
        
     You can interpret as how many times the correlation between the unobservable confounder and the treatment is to
     that between the benchmark confounder and the treatment. The default value is 1.

   :param int or list k_y: 
     :math:`R^2` between result y and the unobservable confounder / :math:`R^2` between result y and the benchmark confounder
        
     You can interpret it as how many times the correlation between the unobservable confounder and the result is to that between the benchmark confounder and the result. The default value is 1.

   :param int threshold: 
        The threshold level in the result plot. The default value is 0.

   :param str measure: 
        The measure you want to shown in the result plot. You can choose between 'att' and 't'. The default measure is 'att'.
        'att': The estimated average treatment effect.
        't': The t-value of the estimated average treatment effect.

   .. method:: plot_result(self, lim_x: float = 0.8, lim_y: float = 0.8, threshold: int or float = 0)
      
      Method for omiited variable bias based sensitivity analysis.
      
      The result plot presents how the averaged treatment effect or the t-value will change with different values of :math:`R^2` between unobservable confounder and T and that between unobservable confounder and Y.

      :param float lim_x: 
         x-axis limit of the plot.

      :param float lim_y: 
         y-axis limit of the plot.
