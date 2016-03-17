# A/B Testing: Udacity’s Free Trial Screener

## Experiment Design

### Metric Choice

>Which of the following metrics would you choose to measure for this experiment
and why? For each metric you choose, indicate whether you would use it as an
invariant metric or an evaluation metric.

* **Invariant metrics:** Number of cookies, Number of clicks, Click-through-probability
* **Evaluation metrics:** Gross conversion, Net conversion

* **Number of cookies:** Suitable invariant metric because the visits happen before the user
sees the experiment;
* **Number of user-ids:** Not a suitable invariant metric because the number of users who enroll
in the free trial is dependent on the experiment;
* **Number of clicks:** Suitable invariant metric because the visits happen before the user
sees the experiment;
* **Click-through-probability:** Suitable invariant metric because the visits happen before the user
sees the experiment;
* **Gross conversion:** Not a suitable invariant metric because the number of users who enroll in the
free trial is dependent on the experiment. Suitable evaluation metric because it is dependent on the
effect of the experiment and allows us to show whether we managed to decrease the cost of enrollments
that aren’t likely to become paying customers.
* **Retention:** Not a suitable invariant metric because the number of users who enroll in the free
trial is dependent on the experiment. Suitable evaluation metric because it is directly dependent on
the effect of the experiment.
* **Net conversion:** Not a suitable invariant metric because the number of users who enroll in the
free trial is dependent on the experiment. Suitable evaluation metric because it is directly dependent o
n the effect of the experiment.


>You should also decide now what results you will be looking for in order to
launch the experiment. Would a change in any one of your evaluation metrics be
sufficient? Would you want to see multiple metrics all move or not move at the
same time in order to launch? This decision will inform your choices while
designing the experiment.

**Gross conversion** and **Net conversion** are the metrics I will focus on. The first metric will
show whether costs are lowered by introducing the screener. The second metric will show how the
change affects our revenues.

To launch the experiment, I require **Gross conversion** to have a statistically significant decrease,
and **Net conversion** to have a statistically significant increase.


### Measuring Variability

To evaluate whether the analytical estimates of standard deviation are accurate
and matches the empirical standard deviation, the unit of analysis and unit of
diversion are compared for each evaluation metric. Assuming Bernoulli distribution
with probability `p` and population `N`, the standard deviation is given by
`sqrt(p*(1-p)/N)`.

As **Gross conversion** and **Net conversion** have the number of cookies as
their denominator, i.e., the unit of diversion. Therefore, the analytical
estimate are expected to be accurate.

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

#### Number of Samples vs. Power

>Indicate whether you will use the Bonferroni correction during your analysis phase,
and give the number of pageviews you will need to power you experiment appropriately.
(These should be the answers from the "Calculating Number of Pageviews" quiz.)

I did not use the Bonferroni correction.

The evaluation metrics I selected to proceed with are *Gross conversion* and *Net conversion*.

We need 685,324 pageviews to power the experiment with these metrics (alpha = 0.05 and beta = 0.2).

#### Duration vs. Exposure

>Indicate what fraction of traffic you would divert to this experiment and, given this,
how many days you would need to run the experiment. (These should be the answers from
the "Choosing Duration and Exposure" quiz.)

I would divert 70% of the traffic to the experiment. The experiment will then take
25 days. This seems to be reasonable.

### Analysis


### Follow-Up Experiment: How to Reduce Early Cancellations
