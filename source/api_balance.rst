Balance Checking
----

.. py:class:: CEM_LinearInf.balance(df_match: pd.DataFrame, df_all: pd.DataFrame, confounder_cols: list, cont_confounder_cols: list, col_y: str = 'Y', col_t: str = 'T')

   Class of balance assessment for the matched data.

   When we finish the coarsened exact matching, it is necessary to evaluate the quality of the matching with imbalance checking methods. 
   
   When the covariate balance is achieved, the resulting effect estimate is less sensitive to model misspecification and ideally close to true treatment effect (Greifer, 2023).

   The imbalance checking methods provided include:
        '**L1**': Calculate and return the L1 imbalance score.

        '**smd**': Print the standardized mean difference summary table and plots of confounders.

        '**ks**': Plot Kolmogorov-Smirnov Statistics of confounders before and after matching.

        '**density**': Return density plots of confounders before and after matching.

        '**ecdf**': Return empirical cumulative density plots of confounders before and after matching.


   :param pd.Dataframe df_match: 
        The dataframe after matching.

   :param pd.Dataframe df_all: 
        The original dataframe before matching.

   :param list confounder_cols: 
        The column names of confounders among all variables X.

   :param list cont_confounder_cols: 
        The column names of all continuous variables among confounders.

   :param str col_y: 
        The column name of result Y in your dataframe. If not specified, it would be "Y".

   :param str col_t: 
        The column name of treatment T in your dataframe. If not specified, it would be "T".

   .. method:: balance_assessing(self, method: str = 'smd', threshold: list = [0.1, 2], _print: bool = True)
     
      Method for generating the imbalance assessing result.

      :param str method: 
            The method to be used for balance assessment. If it's not specified, the default method is '**smd**'.

               '**L1**': Calculate and return the L1 imbalance score.

               '**smd**': Print the standardized mean difference summary table and plots of confounders.

               '**ks**': Plot Kolmogorov-Smirnov Statistics of confounders before and after matching.

               '**density**': Return density plots of confounders before and after matching.

               '**ecdf**': Return empirical cumulative density plots of confounders before and after matching.

               '**all**': Implement all the methods above.

      :param list threshold: 
            When you choose 'smd' to assess the balance, you can set the balance thresholds for smd and variance ratio.

            If it's not specified, the default thresholds are 0.1 and 2 for standardized mean difference and variance ratio respectively.

      :param bool _print: 
            Whether to print the L1 score.


