LIR Python Likelihood Ratio Library
===================================

This library provides a collection of scripts to aid calibration, and
calculation and evaluation of Likelihood Ratios.

This package follows `sklearn` conventions and relies on `numpy`, `sklearn` and
`matplotlib`.

Contents:

* [Example: a simple score-based LR system](#example-a-simple-score-based-lr-system)
* [Plotting](#plotting)


## Example: a simple score-based LR system

Note that this example is just to illustrate how the LIR package works; it is
by no means realistic.

Let's say that a chocolate bar is missing, and the prosecutor
wants to know whether Bart has eaten it or not. In this case, the forensic
expert would formulate formulate Hypothesis 1 (H1, the prosecutor's hypothesis)
and Hypothesis 2 (H2, the defence's hypothesis):

$$H1: Bart has eaten the chocolate bar$$
$$H2: someone else has eaten the chocolate bar$$

The forensic expert will evaluate the evidence. He observes that Bart has brown
lips and that he has a smile on his face.

$$E: Bart has brown lips and a smile on his face$$

The forensic expert evaluates the evidence and reports a likelihood ratio (LR)
back to the prosecutor. The likelihood ratio is the ratio of two conditional
probabilities: (1) the probability of observing $E$ if $H1$ is true; and (2)
the probability of observing $E$ if $H2$ is true.

$$LR = \frac{P(E | H1)}{P(E | H2)}$$

With the LR and the prior odds, the prosecutor (and the judge) can calculate
the posterior odds, and thus the probability that Bart has eaten the chocolate
bar.

$$posterior\_odds = \frac{P(H1 | E)}{P(H2 | E)}$$ = prior\_odds \cdot LR$$


### Data representation and generation

Let's build an LR system to determine an LR for the hypotheses.
All starts with data. We need to establish the relation between the evidence
and the hypotheses, and we will do it empirically. Let's say we have a database
with observations of people after we gave them chocolate and of people after
not giving them chocolate. These are our reference data. Let's assume that
there is no difference between the observations after secretly eating chocolate
and after giving them chocolate. Of course there is no way to tell whether they
brought a chocolate bar by themselves.

The reference data are initialized as a numpy array:

```py
import numpy as np
X = np.array([
    [1, 1],  # brown lips, smile
    [1, 1],  # brown lips, smile
    [1, 0],  # brown lips, neutral
    [1, -1],  # brown lips, sad
    [0, 1],  # clean lips, smile
    [0, 1],  # clean lips, smile
    [0, 1],  # clean lips, smile
    [0, 0],  # clean lips, neutral
    [0, 0],  # clean lips, neutral
    [0, 0],  # clean lips, neutral
    [0, -1],  # clean lips, sad
    [1, 1],  # brown lips, smile
])
```

The array `X` has twelve rows and two columns. Each row is an instance of an
observation of the two features. The first column indicates brown lips (1) or
clean lips (0). The second column indicates a smile (1), a neutral face (0) or
a sad face (-1).

We also have the labels because for these instances we know whether the
observed person has eaten chocolate (1, corresponding to $H1$) or not (0,
corresponding to $H2$). This is a array with 12 elements:

```py
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
```

### Building an LR system

From the reference data we could create an off-the-shelve classifier from
`sklearn`, but we want a calibrated LR rather than a score from 0 to 1, as most
classifiers do by default. We also need a calibrator.

```py
import lir
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
cal = lir.KDECalibrator()

clf.fit(X, y)
prob = clf.predict_proba(X)[:,1]

lrs = cal.fit_transform(prob, y)
```

Of course the LRs we find this way may suffer from overfitting. In a realistic
case, we would need a training set and a test set to measure performance
adequately.

The above can be rewritten with a `CalibratedScorer`:

```py
import lir
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
cal = lir.KDECalibrator()

scorer = lir.CalibratedScorer(clf, cal)
scorer.fit(X, y)
lrs = scorer.predict_lr(X)
```

The performance of the LR system is typically measured as $C_{llr}$:

```py
cllr = lir.metrics.cllr(lrs, y)
print(cllr)
```

If all is well, this should output the value $0.6485392603044089$.

We could also calculate the LR for Bart eating our chocolate bar or not.

```py
lr = scorer.predict_lr(np.array([[1, 1]]))[0]
print(lr)
```

We will find the value 4.892189102638597, meaning that it is almost 5 times
more likely to observe brown lips and a smile if he has eaten the chocolate
bar.


### Plotting

```py
with lir.plotting.show() as ax:
    ax.pav(lrs, y)

with lir.plotting.show() as ax:
    ax.ece(lrs, y)

with lir.plotting.show() as ax:
    ax.lr_histogram(lrs, y)

with lir.plotting.show() as ax:
    ax.tippett(lrs, y)

with lir.plotting.show() as ax:
    ax.score_distribution(clf.predict_proba(X)[:,1], y)
    ax.calibrator_fit(cal)
```


### Summary

```py
import lir
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [1, 1],  # brown lips, smile
    [1, 1],  # brown lips, smile
    [1, 0],  # brown lips, neutral
    [1, -1],  # brown lips, sad
    [0, 1],  # clean lips, smile
    [0, 1],  # clean lips, smile
    [0, 1],  # clean lips, smile
    [0, 0],  # clean lips, neutral
    [0, 0],  # clean lips, neutral
    [0, 0],  # clean lips, neutral
    [0, -1],  # clean lips, sad
    [1, 1],  # brown lips, smile
])

y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

clf = LogisticRegression()
cal = lir.KDECalibrator()

scorer = lir.CalibratedScorer(clf, cal)
scorer.fit(X, y)
lrs = scorer.predict_lr(X)

cllr = lir.metrics.cllr(lrs, y)
print(cllr)

lr = scorer.predict_lr(np.array([[1, 1]]))[0]
print(lr)

with lir.plotting.show() as ax:
    ax.pav(lrs, y)

with lir.plotting.show() as ax:
    ax.ece(lrs, y)

with lir.plotting.show() as ax:
    ax.lr_histogram(lrs, y)

with lir.plotting.show() as ax:
    ax.tippett(lrs, y)

with lir.plotting.show() as ax:
    ax.score_distribution(clf.predict_proba(X)[:,1], y)
    ax.calibrator_fit(cal)
```


## Plotting

Various ways to generate a PAV plot, with different levels of control.

```py
import matplotlib.pyplot as plt

# show a PAV plot on screen
with lir.plotting.show() as ax:
    ax.pav(lrs, y)

# write it to file with a custom title
with lir.plotting.savefig("pav1.png") as ax:
    ax.pav(lrs, y)
    ax.title("PAV plot using savefig()")

# both on screen and to file
with lir.plotting.axes() as ax:
    ax.pav(lrs, y)
    ax.savefig("pav2.png")
    ax.show()

# simple call with more control
fig = plt.figure()
lir.plotting.pav(lrs, y)
plt.savefig("pav3.png")
plt.close(fig)

# with the option to use sub plots
fig, axs = plt.subplots(2)
lir.plotting.pav(lrs, y, ax=axs[0])
lir.plotting.ece(lrs, y, ax=axs[1])
plt.show()
plt.close(fig)
```

The package offers a range of plotting functions.

```py
# Pool adjacent violators (PAV)
with lir.plotting.show() as ax:
    ax.pav(lrs, y)

# Empirical cross entropy (ECE)
with lir.plotting.show() as ax:
    ax.ece(lrs, y)

# Tippett
with lir.plotting.show() as ax:
    ax.tippett(lrs, y)

# Normablized bayes error
with lir.plotting.show() as ax:
    ax.nbe(lrs, y)

# Histogram of LRs
with lir.plotting.show() as ax:
    ax.lr_histogram(lrs, y)

# score distribution
with lir.plotting.show() as ax:
    ax.score_distribution(scores, y)

# calibrator fit
with lir.plotting.show() as ax:
    ax.calibrator_fit(cal)

# score distribution and calibrator fit
with lir.plotting.show() as ax:
    ax.score_distribution(prob, y)
    ax.calibrator_fit(cal)
```
