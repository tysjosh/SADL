"""Reference implementation of the experiments in

  "Sequential Adversarial Distinction Learning: Environment-Invariant
   Pretraining for Out-of-Distribution Generalization"

Layout:
  sadl.data      multi-environment datasets with known generative factors
  sadl.models    shared encoder family, separator heads, Confuser
  sadl.methods   SADL and every baseline of Table 2
  sadl.eval      metrics of Section 6.4 and the statistics of Section 6.5
  sadl.experiments  RQ1-RQ6 runners and the table builders of Section 6.6
"""

__version__ = "0.1.0"
