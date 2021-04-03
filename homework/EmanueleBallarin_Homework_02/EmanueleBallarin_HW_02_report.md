# *Slipping* on the *Shoulders of Giants*? (*HW02*)

### Deep Learning Course $\in$ DSSC @ UniTS (Spring 2021)

#### Submitted by [Emanuele Ballarin](mailto:emanuele@ballarin.cc)  

The following document is a brief, informal report trying to address the last *three* points of proposed [*Homework \#2*](https://github.com/ansuini/DSSC_DL_2021/blob/main/labs/02-sgd-training.ipynb), being it separate from the remaining notebooks as to allow for a more broadly-scoped, free-form analysis of what has been done and the potential insights gained in doing so.

#### The *fully-compliant* approach

In trying to stick as much as possible to the [original paper](https://sci-hub.do/10.1038/323533a0) and the exact requests of the *homework*, the dataset construction, network architecture, loss function, training protocol, batch-size and training duration proposed by *Rumelhart et al., 1986* have been thoroughly followed. As requested per the *homework*, *vanilla GD* has been used as the optimizer of choice.  

To address the only vaguely-specified point, i.e. the choice of nonlinearities and output function, `Sigmoid` activation has been chosen for both hidden and output units.  

The just-described approach leads to successful and rapidly-converging learning for the initial *~200* (of the 1450 total) epochs, after which a *knee-point* for the loss function on the training set is reached, and after which it plateaus.  

Correspondingly, by continuously evaluating test-set accuracy after each epoch to assess generalization, after an eventual noisy transient, accuracy stabilizes as the loss function *kneels* and rarely (across different realizations of the training process) further decreases as training proceeds (even considering >10000 epochs time horizon).  

The issue with that, however, accuracy converges toward *~0.875* and the resulting output is fixed at constantly-zero after discretization of the final `Sigmoid` (meaning: classifying all inputs as not showing mirror symmetry).  

The most plausible explanation of such phenomenon lies in the large imbalance -- across the entire dataset/batch -- between the two classes ("mirror" and "not mirror") in the number of examples present, biasing the training towards higher relevance of the "not mirror" class and the observed result as a consequence.

Anecdotal massive re-simulation of the process showed a convergence-to-*1.0*-accuracy rate of less than $\frac{1}{90}$, potentially exhibiting strong *weight-initialization-dependence* of the network.  

Parameter inspection of the learned model showed convergence at solutions very different from that reported in the original paper (in the average case) and solutions reminiscent of the one in the original paper in the converging cases.  

Anecdotal reports of inspection of the initial weights may allow to constrain the successfully-converging-at-*1.0*-accuracy initial weights inside an $\epsilon$-ball obtained via clustering.

##### Incremental extensions

In trying to alleviate the problems just described, the following incremental modifications upon the original have been tried (and their result reported in braces). The following list is strongly distilled and in no way exhaustive. Sadly, most of the data originating from such experimentations have not been conserved, but are easily reproducible.

- Include hard-thresholding of network output during training, and back-propagate through it (no difference);
- Include hard-thresholding of NN hidden and output units during training, and back-propagate through it (decrease in average accuracy);
- Use of `Tanh` activation for hidden units (faster convergence, no difference in accuracy)
- Use hard, almost-everywhere-differentiable activation functions for hidden units, e.g. `Hard Sigmoid`, `Hard Tanh` (no difference);
- Use higher *LR*, e.g. ~0.5 (no difference when converging, increased risk of divergence);
- Use the `RMSProp` optimizer (no difference);
- Use the exact same hyper-parameters as in the original paper, including momentum (no difference);
- Use the `RMSProp` optimizer in a *high-LR, low-momentum* regime, e.g. `lr=0.8`, `mom=0.1` (increased rate of convergence at *1.0* accuracy; approx. $\frac{1}{25}$).

#### The *re-balancing/re-weighting* approach

Being it of no difference (as explained right above) to use the original hyper-parameters in the paper and *momentum-less, vanilla GD*, the former has been always chosen (except if otherwise noted), in an effort to reproduce the original results.

One possible, more radical, way to workaround the problem is to re-balance the dataset and train on such a rebalanced version of it the same network(s) as above.

*Full-dataset GD*  being employed, both actual *re-balancing via oversampling the least represented class* and *re-weighting the loss based on the inverse of the cardinality of the target class* are equivalent and lead to the same results. Both have been experimented with, finally settling with the re-weighting as the one producing the terser code.

In order to study the long-term-horizon behaviour of the loss, training length has been extended to 10000 epochs. All convergences (if happening, described below) lead to a *plateau* stably establishing between the *2000*th and the *3000*th epoch.

By achieving *perfect re-balance* (obtaining equal relevance of the two classes, i.e. weighting the "mirror" class 7 times the other), no overall change in the training dynamics is observed, except for:

- A higher rate of *failed runs* with accuracy lower than *0.875*;
- A slight increase in runs converging at e.g. *~0.9* accuracy, with some of the "mirror" class inputs being correctly classified;
- A very slight increase in runs converging at e.g. *~0.5* accuracy, with all "mirror" class examples being correctly classified;
- A slight increase in convergent-at-*1.0*-accuracy runs rate (approx. $\frac{1}{20}$).

However, an interesting *transition* is encountered if over-sampling the "mirror" class even more (i.e. *8* times the other). Indeed, a moderate increase in convergent-at-*1.0*-accuracy runs rate (approx. $\frac{1}{6}$) is observed, at the cost of some more *~0.5* accuracy convergent runs with all "mirror" examples correctly classified.

By direct parameter inspection of such latest successfully-converging cases, a parameter setup closer to that reported in the paper can be seen. However, the impression is that we are still far from the *crispness* and elegance of the exact solution shown by *Rumelhart et al.*.

Anecdotal experiments with previously-successful `RMSProp` show an increase in convergence issues, with hardly controllable behaviour.

#### One, last *desperate* and *serendipitous* trial

Reminiscent of the difficulties encountered until now (and maybe, after all, expected?) in such low-data, minimal-parametrization regime, lacking even the hope for Terrence Sejnowsky's *"There's always at least one gradient's coordinate pointing toward an almost-global minimum"* conjecture, this last trial just wanted to assess if -- after all -- even what would be very recent approaches to the problem would have still been powerless.  

The approach built upon the original *adherent-to-the-original*, with just two modifications:

- the use of default *Gaussian with layer-width-adaptive variance* for weight initialization (in any case ineffective in previous trials);
- the use of the `Linear-Sigmoidal` hidden nonlinearity (a.k.a. `SiLU` or `Swish`): $\text{SiLU}(x) := x\sigma(x)$.

Apart from wide acclaim and wildly positive results in some notoriously hard tasks in highly-overparameterized, low- and high- data regimes, the intriguing idea of the `SiLU` is its resemblance to the `ReLU`, but with the possibility to exploit three activation *regimes*:

- the positive-linear ($x, \text{ as } x \rightarrow +\infty$), asymptotically reached with very slight gap;
- the negative-suppressive ($0, \text{ as } x \rightarrow -\infty$) reached from $0^{-}$;
- the bounded-inhibitory ($x < 0 \text{ as } x \text{ is close to } 0^{-}$).

Intuitively, this triple possibility seems to suggest -- with careful bias tuning -- the ability to reach an activation pattern (even if hidden units are only two) that is both dependent from weight initialization and leading to high accuracies. A behaviour already observed in meta-stable or multi-stable learning systems recently proposed. Further investigation is however needed to verify such claim(s).

Results of such trial have been thrilling, with still no run diverged, approximately $\frac{1}{100}$ runs converged at accuracy *0.875* and all remaining converging stably at *1.0*, while test-set accuracy step-wise increasing with epoch. Maximum accuracy was always reached between *4000*th and *5000*th epoch.  

Direct parameter inspection showed no clear resemblance of successfully-converged runs with the results shown in the original paper. This phenomenon, however must be seen in the light of a different and much more *irregular* activation function being used for hidden units.