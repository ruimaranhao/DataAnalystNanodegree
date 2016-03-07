# A/B Testing: Udacity’s Free Trial Screener

## Experiment Design

### Metric Choice

>Which of the following metrics would you choose to measure for this experiment
and why? For each metric you choose, indicate whether you would use it as an
invariant metric or an evaluation metric.

* **Invariant metrics:** Number of cookies, Number of clicks, Click-through-probability
* **Evaluation metrics:** Gross conversion, Net conversion


>You should also decide now what results you will be looking for in order to
launch the experiment. Would a change in any one of your evaluation metrics be
sufficient? Would you want to see multiple metrics all move or not move at the
same time in order to launch? This decision will inform your choices while
designing the experiment.




### Measuring Variability

To evaluate whether the analytical estimates of standard deviation are accurate
and matches the empirical standard deviation, the unit of analysis and unit of
diversion are compared for each evaluation metric. Assuming Bernoulli distribution
with probability `p` and population `N`, the standard deviation is given by
`sqrt(p*(1-p)/N)`.

As **Gross conversion** and **Net conversion** have the number of cookies as
their denominator, i.e., the unit of diversion, the analytical estimate are
expected to be accurate.

#### Gross conversion

```
p = 0.20625
N = 5000 * 0.08 = 400
stdev = sqrt(0.20625 * (1-0.20625) / 400) = 0.0202
```

#### Net conversion
```
p = 0.1093125
N = 5000 * 0.08 = 400
std dev = sqrt(0.1093125 * (1-0.1093125) / 400) = 0.0156
```

### Sizing



### Analysis


### Follow-Up Experiment: How to Reduce Early Cancellations
