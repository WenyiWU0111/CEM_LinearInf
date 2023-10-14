Treatment Effect Inference 
---------------------------

.. py:class:: inference(df: pd.DataFrame, col_y: str = 'Y', col_t: str = 'T', col_x: list = None, confounder_cols: list = None)

   Class of linear inference methods for estimating average treatment effect and heterogeneous treatment effect.

   After conducting the coarsened exact matching and imbalance checking, we can estimate the average treatment effect ATT and heterogeneous treatment effect HTE with statistical inference methods.

   Ordinal least square linear regression and weighted least square linear regression methods are provided for the ATT estimation, and the linear double machine learning method is provided for the HTE estimation.
   
   :param pd.DataFrame df: 
        The dataframe after matching.

   :param str col_y: 
        The column name of dependent variable Y in your dataframe. If not specified, it would be "Y".

   :param str col_t: 
        The column name of treatment variable T in your dataframe. If not specified, it would be "T".

   :param list col_x: 
        A list of column names of control variables X in your dataframe, which must be specified.
        
        Names of confounders should not be included in this list.

   :param list confounder_cols: 
        A list of column names of confounders W in your dataframe, which must be specified.

   .. method:: linear_att()

      Method for estimating the ATT with the ordinal least square linear model.
      
      Return the estimated ATT and print out the model summary.

   .. method:: weighted_linear_att()

      Method for estimating the ATT with the weighted least square linear model.
      
      Return the estimated ATT and print out the model summary.

   .. method:: linear_dml_hte(self, final_model: str = 'ols_linear')
      
      Method for estimating the HTE with the linear double machine learning method.
      
      Return the average treatment effect on treated ATT, conditional average treatment effect CATE, R2 score of the model.

      :param str final_model: You can choose among ['ols_linear', 'lasso', 'ridge'] as your second stage model.

      :returns:
       float: conditional average treatment effect on treated CATE
       
       list: heterogeneous treatment effect HTE, 
       
       float: R2 score of the model

