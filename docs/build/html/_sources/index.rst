.. CEM_LinearInf documentation master file, created by
   sphinx-quickstart on Mon Aug 21 12:59:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=========================================
CEM-LinearInf's Documentation
=========================================

.. raw:: html

   <div style="border-top: 4px solid gray;"></div>


**CEM-LinearInf** is a Python package for linear causal inference, which can help you implement the whole process of causal inference easily. 

Please check out the following sections for further information, including how to
:ref:`install <installation>` and :ref:`use <tutorial>` this package, and :ref:`related references <references>`.

✏️ Functions
------------
* Coarsened Exact Matching (CEM)
   CEM is a data preprocessing algorithm in causal inference which can construct your observational data into 'quasi' experimental data easily, mitigating the model dependency, bias, and inefficiency of your estimation of the treatment effect (:ref:`Ho, Imai, King, & Stuart 2007<ref3>`).
   
   Different coarsen methods and 1 to k matching method based on different distances are supported.

* Balance Checking
   When we finish the coarsened exact matching, it is necessary to evaluate the quality of the matching with balance checking methods. When the covariate balance is achieved, the resulting effect estimate is less sensitive to model misspecification and ideally close to true treatment effect (Greifer, 2023).  

   Different methods including L1 imbalance score, SMD, KS score, density plot, and empirical cdf plot are supported.

* Treatment Effect Inference
   After conducting the coarsened exact matching and imbalance checking, we can estimate the average treatment effect **ATT** and heterogeneous treatment effect **HTE** with statistical inference methods.

   **Linear regression** models including OLS, Ridge, and Lasso are supported here.

* Sensitivity Analysis
   When we conduct causal inference to the observational data, the most important assumption is that there is no unobserved confounding. Therefore, after finishing the treatment effect estimation, investigators are advised to conduct the sensitivity analysis to **examine how fragile a result is against the possibility of unobserved confounders** (:ref:`Cinelli, Hazlett, 2020<ref2>`).  
   
   In other words, we should examine how strong the effect of unobserved confounders should be to erase the treatment effect estimated.
   
   **Omitted variable bias** based sensitivity analysis method and **Wilcoxon's  signed rank test** based sensitivity analysis method are supported here.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

📖 Contents
-----------

.. toctree::

   usage
   tutorial
   example
   api


.. _references:

⭐️ Reference
-------------

.. _ref1:

* Abadie, A., & Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects. Econometrica, 74(1), 235–267. https://doi.org/10.1111/j.1468-0262.2006.00655.x 

.. _ref2:

* Cinelli, C., & Hazlett, C. (2020). Making Sense of Sensitivity: Extending Omitted Variable Bias. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(1), 39–67. https://doi.org/10.1111/rssb.12348 

.. _ref3:

* Ho, D., Imai, K., King, G., & Stuart, E. (2007). Matching as Nonparametric Preprocessing for Reducing Model Dependence in Parametric Causal Inference. Political Analysis, 15, 199–236. Retrieved from https://tinyurl.com/y4xtv32s 

.. _ref4:

* Iacus, S. M., King, G., & Porro, G. (2009). CEM: Coarsened Exact Matching Software. Journal of Statistical Software, 30(9). Retrieved from http://gking.harvard.edu/cem 

.. _ref5:

* Iacus, S. M., King, G., & Porro, G. (2012). Causal Inference Without Balance Checking: Coarsened Exact Matching. Political Analysis, 20(1), 1–24. Retrieved from https://tinyurl.com/yydq5enf 


