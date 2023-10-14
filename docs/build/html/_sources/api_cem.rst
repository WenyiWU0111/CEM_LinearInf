Coarsened Exact Matching
-------------------------

.. py:class:: CEM_LinearInf.cem(df: pd.DataFrame, confounder_cols: list, cont_confounder_cols: list, col_y: str = 'Y', col_t: str = 'T')

   Class of Coarsened Exact Matching (CEM).

   CEM is a data preprocessing algorithm in causal inference that has broad applicability to observational data.
   With CEM, you can construct your observational data into 'quasi' experimental data easily, mitigating the model dependency, bias, and inefficiency of your estimation of the treatment effect (King and Zeng 2006; Ho, Imai, King, and Stuart 2007; Iacus et al. 2008).

   :param pd.Dataframe df: The dataframe you want to summarize.

   :param list confounder_cols: The column names of confounders among all variables X.

   :param list cont_confounder_cols: The column names of all continuous variables among confounders.

   :param string col_y: The column name of result Y in your dataframe. If not specified, it would be "Y".

   :param string col_t: The column name of treatment T in your dataframe. If not specified, it would be "T".

   .. method:: summary()
      
      Method for generating the summary of data, return the descriptive statistics result of the dataframe
      and the T-test result of Experimental group `Y_1` and Control group `Y_2`.

   .. method:: cut(col: pd.Series, func: str, param: int or list = None)

      Method for cutting a continuous confounder X into discrete bins.

      :param pd.Series col: Continuous confounder X to be coarsened.

      :param str func: Cutting function to be used. The following functions can be chosen.

            '**cut**': Bin values into discrete intervals with the same length.
            
            '**qcut**': Discretize variable into equal-sized buckets based on rank or based on sample quantiles.
            
            '**struges**': Bin values into discrete intervals with the same length k according to the Sturges' rule.

      :param int or list (optional) param:
            
            When the method is '**cut**', it's a number of bins or a list of bins' edges;
            
            When the method is '**qcut**', it's a number of quantiles or a list of quantiles, *e.g.* [0, .25, .5, .75, 1.] for quartiles.
            
            When the method is '**struges**', there's no need to specify the param.

      :returns: pd.Series: Coarsened confounder X.

   .. method:: coarsening(self, coarsen_df: pd.DataFrame = None, cont_confounder_cols: list = None, schema: dict = {})

      Coarsen your data based on the specified coarsen schema. If the coarsen schema is not specified, the default method is binning values into discrete intervals with the same length **k** according to the Sturges' rule.

      :param pd.DataFrame coarsen_df: DataFrame to be coarsened.

      :param list cont_confounder_cols: Column names of continuous confounders of the dataframe to be coarsened.

      :param dict chema: The dictionary specifing what coarsening method you want to use on each continuous confounder **X**.
            
            If it is not specified, the default method is binning values into discrete intervals with the same length k according to the Sturges' rule.

      :returns: pd.DataFrame: Dataframe with coarsened confounders **X**.

   .. method:: weight(self, strata_T: pd.Series, all_T: pd.Series) 

      Method of computing weights for each observation.

      :param pd.Series strata_T: The treatment **T** of samples in one strata.

      :param  pd.Series all_T: The treatment **T** of all the matched samples.

      :returns: pd.Series: Weights of samples in the input strata.

   .. method:: match(self, schema: dict = {}, k2k_ratio: int = 0, dist: str = 'euclidean', _print: bool = True)

      Method for perform coarsened exact matching using the specified coarsening schema and return the dataframe with weights for each observation.
      
      If the coarsen schema is not specified, the default method is binning values into discrete intervals with the same length k according to the Sturges' rule.

      :param dict schema: 
            
            The dictionary specifing what coarsening method you want to use on each continuous confounder **X**.
            
            If it is not specified, the default method is binning values into discrete intervals with the same length k according to the Sturges' rule.

      :param int k2k_ratio: 
            
            The ratio of (# control samples / # treatment samples) when we conduct k to k matching.
            
            If `k2k_ratio` is 0, we don't conduct k to k matching.
            
            Otherwise, all treatment samples in each strata will be matched with k nearest control samples.

      :param str dist: The measure of distance when conducting k to k matching.
            
            You can choose among ['euclidean', 'mahalanobis;, 'psm']. The default one is '**euclidean**'.

      :param bool _print: Whether to print the matching result.

      :returns: pd.DataFrame: The matched dataframe.

   .. method:: k2k_match(self, df: pd.DataFrame, confounder_cols: list, group: list, dist: str, k2k_ratio: int = 1)

      Method for performing k to k coarsened exact matching using the specified distance measure, and return the dataframe with weights for each observation.

      :param pd.DataFrame df: 
            The coarsened dataframe to be matched.

      :param  list confounder_cols: 
            The column names of all continuous variables among confounders.

      :param list group: 
            The column names of variables used for grouping.
            
            It's equivalent to dis_confounder_cols + [\'coarsen_\' + name for name in cont_confounder_cols].

      :param str dist: 
            The measure of distance when conducting k to k matching.
            
            You can choose among ['euclidean', 'mahalanobis;, 'psm']. The default one is '**euclidean**'.

      :param int k2k_ratio: 
            The ratio of (# control samples / # treatment samples) when we conduct k to k matching.
            
            All treatment samples in each strata will be matched with k nearest control samples.
            
            The default ratio is **1**.

      :returns: 
         
         pd.DataFrame: The matched dataframe.
         
         dict: A dictionary of pair index, indicating which control sample is paired with each experimantal sample.

   .. method:: tunning_schema(self, step: int = 4)

      Method for fine-tunning continuous confounders' binning schema automatically. The optimization objective is to have a smaller L1 score conditional on a relatively large matched sample size.

      :param int step: The step when tunning the number of bins.
            
         E.X. When 'step' = 4, the function will calculate all L1 scores when the number of bins equals ::math:`[4, 8, ..., 1 + \log_{2}n]`.

      :returns: 
      
         pd.DataFrame: A dataframe recording L1 scores of each schemas.
         
         dict: The optimal schema dictionary.

