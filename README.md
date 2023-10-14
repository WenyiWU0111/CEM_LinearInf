# CEM_LinearInf

<div style="border-top: 4px solid gray;"></div>


**CEM-LinearInf** is a Python package for linear causal inference, which can help you implement the whole process of causal inference easily.

Please check out the following sections for further information, including how to install and use this package, and related references.

- Documentation: []()
- GitHub repo: []()
- PyPi: []()
- Open Source: [MIT license](https://opensource.org/licenses/MIT)

## 💡 Installation

Install `CEM_LinearInf` using ``pip``

```bash
pip install CEM_LinearInf
```

Install the latest version in Github:
```bash
pip install git+
```

## 📖 Examples and notebooks

- [CEM Notebook Example - Simulated Linear Dataset](CEM_Notebook_Example_Lalonde_Dataset.ipynb)
- [CEM Notebook Example - Lalonde Dataset]()
- [Tutorial]()

## ✏️ Functions

* Coarsened Exact Matching (CEM)

   CEM is a data preprocessing algorithm in causal inference which can construct your observational data into 'quasi' experimental data easily, mitigating the model dependency, bias, and inefficiency of your estimation of the treatment effect (Ho, Imai, King, & Stuart 2007).
   
   Different coarsen methods and 1 to k matching method based on different distances are supported.

* Balance Checking

   When we finish the coarsened exact matching, it is necessary to evaluate the quality of the matching with balance checking methods. When the covariate balance is achieved, the resulting effect estimate is less sensitive to model misspecification and ideally close to true treatment effect (Greifer, 2023).  

   Different methods including L1 imbalance score, SMD, KS score, density plot, and empirical cdf plot are supported.

* Treatment Effect Inference

   After conducting the coarsened exact matching and imbalance checking, we can estimate the average treatment effect **ATT** and heterogeneous treatment effect **HTE** with statistical inference methods.

   **Linear regression** models including OLS, Ridge, and Lasso are supported here.

* Sensitivity Analysis

   When we conduct causal inference to the observational data, the most important assumption is that there is no unobserved confounding. Therefore, after finishing the treatment effect estimation, investigators are advised to conduct the sensitivity analysis to **examine how fragile a result is against the possibility of unobserved confounders** (Cinelli, Hazlett, 2020).  
   
   In other words, we should examine how strong the effect of unobserved confounders should be to erase the treatment effect estimated.
   
   **Omitted variable bias** based sensitivity analysis method (Cinelli, Hazlett, 2020) and **Wilcoxon's  signed rank test** based sensitivity analysis method (Rosenbaum, 2015) are supported here.


## ⭐️ Reference

- Abadie, A., & Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects. Econometrica, 74(1), 235–267. https://doi.org/10.1111/j.1468-0262.2006.00655.x

-  Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2017). Double/debiased machine learning for treatment and causal parameters.

-  Cinelli, C., & Hazlett, C. (2020). Making Sense of Sensitivity: Extending Omitted Variable Bias. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(1), 39–67. https://doi.org/10.1111/rssb.12348

-  Cinelli, C., & Ferwerda, J., & Hazlett, C. (2020). “sensemakr: Sensitivity Analysis Tools for OLS in R and Stata.” https://www.researchgate.net/publication/340965014_sensemakr_Sensitivity_Analysis_Tools_for_OLS_in_R_and_Stata

-  Greifer N (2023). cobalt: Covariate Balance Tables and Plots. https://github.com/ngreifer/cobalt.

-  Ho, D., Imai, K., King, G., & Stuart, E. (2007). Matching as Nonparametric Preprocessing for Reducing Model Dependence in Parametric Causal Inference. Political Analysis, 15, 199–236. Retrieved from https://tinyurl.com/y4xtv32s

-  Ho D, Imai K, King G, Stuart E (2011). “MatchIt: Nonparametric Preprocessing for Parametric Causal Inference.” Journal of Statistical Software, 42(8), 1–28. https://doi.org/10.18637/jss.v042.i08.

-  Iacus, S. M., King, G., & Porro, G. (2009). CEM: Coarsened Exact Matching Software. Journal of Statistical Software, 30(9). Retrieved from http://gking.harvard.edu/cem

-  Iacus, S. M., King, G., and Porro, G. (2011). Multivariate Matching Methods That are Monotonic Imbalance Bounding. Journal of the American Statistical Association, 106(493), 345-361. Retrieved from https://tinyurl.com/y6pq3fyl

-  Iacus, S. M., King, G., & Porro, G. (2012). Causal Inference Without Balance Checking: Coarsened Exact Matching. Political Analysis, 20(1), 1–24. Retrieved from https://tinyurl.com/yydq5enf

-  Rosenbaum, P. R. (2005). Sensitivity analysis in observational studies. Encyclopedia of statistics in behavioral science.

-  Standardized mean difference (SMD) in causal inference. (2021, Oct 31). Retrieved from https://statisticaloddsandends.wordpress.com/2021/10/31/standardized-mean-difference-smd-in-causal-inference/

-  What is the L1 imbalance measure in causal inference? (2021, Nov 25). Retrieved from https://statisticaloddsandends.wordpress.com/2021/11/25/what-is-the-l1-imbalance-measure-in-causal-inference/

-  Zhang, Z., Kim, H. J., Lonjon, G., Zhu, Y., & written on behalf of AME Big-Data Clinical Trial Collaborative Group (2019). Balance diagnostics after propensity score matching. Annals of translational medicine, 7(1), 16. https://doi.org/10.21037/atm.2018.12.10
