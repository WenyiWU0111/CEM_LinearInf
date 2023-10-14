Sensitivity Analysis
==========================

When we conduct causal inference to the observational data, the most important assumption is that there is no unobserved confounding.  
Therefore, after finishing the treatment effect estimation, investigators are advised to conduct the sensitivity analysis to **examine how fragile a result is against the possibility of unobserved confounders** (:ref:`Cinelli, Hazlett, 2020<ref2>`).  
In other words, we should examine how strong the effect of unobserved confounders should be to erase the treatment effect estimated. 

Two methods are provided in our package, including  **Omitted variable bias** based sensitivity analysis method (:ref:`Cinelli, Hazlett, 2020<ref2>`) and **Wilcoxon's  signed rank test** based sensitivity analysis method (:ref:`Rosenbaum, 2015<ref1>`).

1️⃣ OVB 
-----------------------------------------

.. note:: 

   This method can be used if your result variable :math:`Y` is linearly dependent with :math:`X` and :math:`T`.

The **Omitted variable bias** based sensitivity analysis result gives us the following informations, and all of these informations can be found in only one picture.

*   **Robustness Value (RV)**:  

The robustness value represents the threshold level of association that unobserved confounder must reach, both with the treatment and the outcome, in order to alter the conclusions of the research.

The result figure provides a convenient reference point to assess the overall robustness of a coefficient to unobserved confounders. If the confounder's association to the treatment :math:`R_{Y\sim Z|T, X}^2` and to
the outcome :math:`R_{Z\sim T|X}^2` are both assumed to be less than the :math:`RV`, then such confounders cannot “explain away” the observed effect.

*   **Contour Line**: 

The points on the same contour line has the same adjusted estimated ATT. The contour line helps us to know the value of the adjusted estimated :math:`ATT` when :math:`R_{Y\sim Z|T, X}^2 = a` and :math:`R_{Z\sim T|X}^2 = b`.

* **Bound the strength of the hidden confounder using observed covariate**:  

We can choose an observed confounder :math:`X_j` as a benchmark, and check the adjusted estimated :math:`ATT` when

.. math::

   \frac{R_{Y\sim Z|T, X_{-j}}^2}{R_{Y\sim X_j|T, X_{-j}}^2} = K_Y 

   \frac{R_{T\sim Z|X_{-j}}^2}{R_{T\sim X_j|X_{-j}}^2} = K_T 

**Advantages**

* Having no parametric assumptions on the distribution of the confounder.

* Having simple sensitivity measures for routine reporting.

* Connecting sensitivity analysis to domain knowledge.

**Limitations**

* Assuming that the unobservable confounder :math:`X` is linearly dependent with result variable :math:`Y` and treatment :math:`T`.

**Example**

Here we choose :math:`X_1` as our benchmark. When :math:`K_Y = K_T = 0.2`, the adjusted estimated :math:`ATT` is 2.9722.

When :math:`R_{Y\sim Z|T, X}^2 = R_{Z\sim T|X}^2 = 0.6803`, the unobserved confounder :math:`X` will “explain away” the observed effect.

.. code-block:: python

   from CEM_LinearInf.sensitivity_analysis import ovb
   import statsmodels.api as sm
   import numpy as np

   X = sm.add_constant(my_cem.matched_df[[my_cem.col_t] + [f'X{i}' for i in range(1, 10)]])
   y = np.asarray(my_cem.matched_df[my_cem.col_y])
   model = sm.WLS(y.astype(float), X.astype(float), weights=1)

   my_ovb = ovb(model=model, bench_variable='X1', k_t = [0.2, 0.5], k_y=[0.2, 0.5],  measure = 'att')
   my_ovb.plot_result()

.. image:: pics/ovb.png
    :align: center
    :alt: smd_result
    :width: 3.03529in
    :height: 2.97222in


2️⃣ Wilcoxon's  signed rank test
--------------------------------

.. note::

   It is suitable for 1-1 matched dataset, which means that only 1 untreated sample are matched with each treated sample, and this can be achieved by setting ``k2k_ratio = 1`` in the ``match`` step.

Wilcoxon's signed rank test based sensitivity analysis imagines that in the population before matching, all samples are assigned to treatment or control independently with unknown probabilities. However, two samples  with the same observed confounders
may nonetheless differ in terms of unobserved confounders, so that one sample has an odds of treatment that is up to :math:`\Gamma ≥ 1` times greater than the odds for another sample.

The sensitivity analysis asks how large the :math:`\Gamma` should be to erase the treatment effect estimated.

With :math:`S` being the number of sample pairs and :math:`W` being the sum of the ranks of the positive differences between pairs, we have

.. math::

   \lambda = \frac{\Gamma}{1 + \Gamma}, \mu_{max} = \frac{\lambda S (S+1)}{2}, \mu_{min} = \frac{(1-\lambda) S (S+1)}{2}, \sigma ^2 = \frac{\lambda(1-\lambda)S(S+1)(2S+1)}{6}

:math:`Z=\frac{X-\mu}{\sigma}` follows the standard normal distribution. If the corresponding :math:`p-value` is greater than 0.05, than we can reject the null hypothesis that
the treatment is randomly assigned, which means that the unobserved confounder erases the treatment effect estimated.

**Example**

In the following example, when :math:`\Gamma = 4.25`, the upper bound of the :math:`p-value`'s interval is greater than 0.05, which means that in this situation, we don't have 95% confidence to reject the null hypothesis that the treatment is randomly assigned. In other words, when :math:`\Gamma = 4.25` the estimated :math:`ATT` will be explained away by unovserved confounders.

.. code-block:: python

   from CEM_LinearInf.sensitivity_analysis import wilcoxon

   my_sen = wilcoxon(df=my_cem_k2k.matched_df, pair = my_cem_k2k.pair)
   wilcoxon_df = my_sen.result([1, 2, 3, 4, 4.25, 5])

.. code-block:: none

          lower_p  upper_p
   gamma                  
   1.00       0.0   0.0000
   2.00       0.0   0.0000
   3.00       0.0   0.0000
   4.00       0.0   0.0112
   4.25       0.0   0.0575
   5.00       0.0   0.6223
   The estimated ATT result is not reliable if there exists an unobservable confounder which makes the magnitude of probability
   that a single subject will be interfered with is 4.25 times higher than that of the other subject.




⭐️ Reference
--------------

.. _ref2:

* Cinelli, C., & Hazlett, C. (2020). Making Sense of Sensitivity: Extending Omitted Variable Bias. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(1), 39–67. https://doi.org/10.1111/rssb.12348 

.. _ref3:

* Cinelli, C., & Ferwerda, J., & Hazlett, C. (2020). “sensemakr: Sensitivity Analysis Tools for OLS in R and Stata.” https://www.researchgate.net/publication/340965014_sensemakr_Sensitivity_Analysis_Tools_for_OLS_in_R_and_Stata

.. _ref1:

* Rosenbaum, P. R. (2005). Sensitivity analysis in observational studies. Encyclopedia of statistics in behavioral science.

.. _tuto_sen:
.. toctree::
   :maxdepth: 2
   :caption: Contents: