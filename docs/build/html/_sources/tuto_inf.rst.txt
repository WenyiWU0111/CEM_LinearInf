Inference
==============

💡 Model
---------

It is important to note that matching methods serve as algorithms for preprocessing data rather than statistical estimators. 
Therefore, once the data has been preprocessed, an estimator of some sort needs to be applied in order to draw causal inferences (:ref:`Iacus, King, & Porro, 2012<inf_ref2>`).

The most frequently selected method following matching has been a basic difference in means, 
neglecting any controls for potential confounding variables. However, unless matching is perfect, conventional parametric approaches have the potential to significantly enhance causal inferences even after the matching process (:ref:`Ho, Imai, King, & Stuart 2007<inf_ref1>`).

Here we provide linear regression methods and linear double machine learning methods for estimating the **average treatment effect ATT** and **heterogeneous treatment effect HTE**.

* Ordinal least square linear regression method ``linear_att`` and weighted least square linear regression method ``weighted_linear_att`` are provided for the ATT estimation.

.. math::

    Y = \hat{\theta}T + \hat{\beta}X + ϵ, \widehat{ATT} = \hat{\theta} 

* Linear double machine learning method ``linear_dml_hte`` is provided for the HTE estimation and conditional average treatment effect estimation CATE. This method was proposed by :ref:`Chernozhukov et al. in 2017<inf_ref3>`. 

.. math::

    Y^{\bot X} = Y - \hat{\beta_{1}}X 

    T^{\bot X} = T - \hat{\beta_{2}}X 

    Y_i^{\bot X} = \widehat{\theta(X_i)}T^{\bot X} + ϵ 

    \widehat{HTE} = \widehat{\theta(X_i)} 

    \widehat{CATE} = E{\widehat{\theta(X_i)}} 

⌨️ Example
-------------

Firstly you should create your own ``inference`` instance, giving it your matched dataframe,  column names of result variable :math:`Y`, treatment variable :math:`T`, control variables :math:`X`, and confounders.

.. code-block:: python

    from CEM_LinearInf.inference import inference

    my_inf = inference(df = my_cem.matched_df, # matched dataframe
                   col_y = 'Y', # column name of result variable
                   col_t = 'T', # column name of treatment variable
                   col_x = ['X4', 'X5', 'X6', 'X8'], # list of column names of control variables, please be noted that confounders should not be included in this list
                   confounder_cols = my_cem.confounder_cols) # list of column names of confounders

**ATT Estimation**

.. code-block:: python

    #att = my_inf.linear_att()
    att = my_inf.weighted_linear_att()
    print(f'att: {round(att, 4)}')

.. code-block:: none 

                                WLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.704
    Model:                            WLS   Adj. R-squared:                  0.704
    Method:                 Least Squares   F-statistic:                     3420.
    Date:                Fri, 21 Jul 2023   Prob (F-statistic):               0.00
    Time:                        10:55:56   Log-Likelihood:                -22394.
    No. Observations:                7194   AIC:                         4.480e+04
    Df Residuals:                    7188   BIC:                         4.484e+04
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -2.5359      0.102    -24.891      0.000      -2.736      -2.336
    T              2.8786      0.146     19.774      0.000       2.593       3.164
    X4            -2.6487      0.059    -45.124      0.000      -2.764      -2.534
    X5             3.6880      0.059     62.155      0.000       3.572       3.804
    X6             3.1113      0.058     53.891      0.000       2.998       3.225
    X8             4.7185      0.052     90.623      0.000       4.616       4.821
    ==============================================================================
    Omnibus:                      481.185   Durbin-Watson:                   1.976
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1254.094
    Skew:                          -0.383   Prob(JB):                    4.75e-273
    Kurtosis:                       4.896   Cond. No.                         5.36
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    att: 2.8786

**HTE Estimation**

.. code-block:: python

    cate, hte, r2 = my_inf.linear_dml_hte()
    #cate, hte, r2 = my_inf.linear_dml_hte(final_model = 'lasso')
    #cate, hte, r2 = my_inf.linear_dml_hte(final_model = 'ridge')
    
    print(f'cate: {round(cate, 4)}, r2:{round(r2, 4)}')

.. code-block:: none 

    cate: 3.0594, r2:0.0007

**Result Comparision**


.. list-table:: Result Table
   :widths: 25 25 25 25 25 25
   :header-rows: 1

   * - 
     - True ATT
     - Diff in Mean 
     - linear_att
     - weighted_linear_att
     - linear_dml_hte
   * - Before Match
     - 3.0
     - 1.0085
     - 0.9117
     - 0.9117
     - 3.06
   * - After Match
     - \
     - \
     - 2.5425
     - 2.8786
     - 3.0594

   
⭐️ Reference
-------------

.. _inf_ref3:

* Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2017). Double/debiased machine learning for treatment and causal parameters.

.. _inf_ref1:

* Daniel Ho, Kosuke Imai, Gary King, and Elizabeth Stuart. (2007). “Matching as Nonparametric Preprocessing for Reducing Model Dependence in Parametric Causal Inference.” Political Analysis, 15, Pp. 199–236. Copy at https://tinyurl.com/y4xtv32s

.. _inf_ref2:

* Stefano M. Iacus, Gary King, and Giuseppe Porro. (2012). “Causal Inference Without Balance Checking: Coarsened Exact Matching.” Political Analysis, 20, 1, Pp. 1--24. Website Copy at https://tinyurl.com/yydq5enf



.. _tuto_inf:
.. toctree::
   :maxdepth: 2
   :caption: Contents: