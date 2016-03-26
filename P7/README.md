# A/B Testing: Udacity’s Free Trial Screener

## Experiment Design

### Metric Choice

>Which of the following metrics would you choose to measure for this experiment
and why?

* **Invariant metrics:** Number of cookies, Number of clicks, Click-through-probability
* **Evaluation metrics:** Gross conversion, Net conversion

>For each metric you choose, indicate whether you would use it as an
invariant metric or an evaluation metric.

* **Number of cookies:**  As unit of diversion is _cookies_, this metric is the invariant
between experiment group and control group.
* **Number of user-ids:** This metric is dependent on the experiment, but
the number if not suitable as evaluation metric because it is not normalized. This metric
neither seems suitable as invariant nor evaluation metric.
* **Number of clicks:** Invariant metric because cookies are randomly assigned to the two
groups and clicks clicks happen before the experiment.
* **Click-through-probability:** As this metric equals the number of clicks divided by the
number of cookies, and those metrics are invariants, this metric is an invariant metric.
* **Gross conversion:** In the experiment group, when people click to start the free trial,
if they indicate that they will not be able to work more than 5 hours per week, they are
suggested not to enroll. Gross conversion is about probability to succeed, and this seems to
be a suitable evaluation metric.
* **Retention:** Not a suitable invariant metric because the number of users who enroll in the free
trial is dependent on the experiment. Suitable evaluation metric because it is directly dependent on
the effect of the experiment.
* **Net conversion:** Not a suitable invariant metric because the number of users who enroll in the
free trial is dependent on the experiment. Suitable evaluation metric because it is directly dependent
on the effect of the experiment.

In summary, the invariant metrics are **number of cookies**, **number of clicks**,
**click-through probability**, and the evaluation metrics are **gross conversion**,
**retention** and **net conversion**. In the experiment, the goal is to reduce the
number of paying people because lack of availability to dedicate to the course, while
not decreasing the number of payments.

I would launch the experiment only when we reduce number of enrollments but do not
reduce number of payments. This is to say that to launch the experiment, I require
**Gross conversion** to have a statistically significant decrease, and
**Net conversion** to have a statistically significant increase. Note that retention
equals net conversion divided by gross conversion.

>You should also decide now what results you will be looking for in order to
launch the experiment. Would a change in any one of your evaluation metrics be
sufficient? Would you want to see multiple metrics all move or not move at the
same time in order to launch? This decision will inform your choices while
designing the experiment.

**Gross conversion**, **Net conversion** and **retention** are the metrics I will
focus on.

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

#### Retention
```
p = 0.53
number of enrollment = 5000 × 0.08 × 0.20625 = 82.5
std dev = sqrt(0.53 * (1-0.53) / 82.5) = 0.0549
```

For **gross conversion** and **net conversion**, their denominators are number of
clicks, which is also unit of diversion, so their analytic variance are likely to
match their empirical variance. For **retention**, its denominator is
**number of enrollments**, which is not the unit of diversion in the experiment,
so its empirical variance would be much higher than analytic variance.

### Sizing

#### Number of Samples vs. Power

>Indicate whether you will use the Bonferroni correction during your analysis phase

I did not use the Bonferroni correction: We want gross conversion significantly
decrease _and_ net conversion to not significantly decrease. Bonferroni
correction is suitable when we are dealing with an _or_ and it is not suitable
when use _and_.

>Give the number of pageviews you will need to power you experiment appropriately.
(These should be the answers from the "Calculating Number of Pageviews" quiz.)

alpha = 0.05; beta = 0.20

* Gross conversion: base conversion rate = 20.625%, dmin = 1%
* Retention: base conversion rate = 53%, dmin = 1%
* Net conversion: base conversion rate = 10.93125%, dmin = 0.75%

Using the [calculator](http://www.evanmiller.org/ab-testing/sample-size.html) referred
in the classes, the number of samples:

* **Gross conversion:** 25835 clicks for each group
* **Retention:** 39115 enrollments for each group
* **Net conversion:** 27413 clicks for each group

Hence, we need:

* **Gross conversion:** 25835 x 40000 / 3200 = 322937.5 pageviews for each group
* **Retention:** 39115 x 40000 x 660 = 2370606 pageviews for each group
* **Net conversion:** 27413 x 40000 / 3200 = 342662.5 pageviews for each group


#### Duration vs. Exposure

>Indicate what fraction of traffic you would divert to this experiment and, given this,
how many days you would need to run the experiment. (These should be the answers from
the "Choosing Duration and Exposure" quiz.)

* Number of pageviews: 2370606 x 2 = 4741212
* fraction: 1.0
* days = 4741212 / 1.0 / 40000 = 119

119 days seems to me way too long for this experiment. A way to reduce the number of
exposure is to consider only **Gross conversion** and **Net conversion**:

* Number of pageviews: 342662.5 x 2 = 685325
* fraction: 1.0
* days = 685325 / 1.0 / 4000 = 18

An experiments that spans 18 days seems adequate. This means that we will not use
**retention** as evaluation metric after all.

The calculations above consider that 100% of the traffic is diverted to the experiment  
(fraction). More days would be needed if we decide to decrease this number, but it
does not seem needed.

### Analysis

#### Sanity Checks

* Control group: number pageviews = 345543 and number clicks = 28378,
* Experiment group: number pageviews = 344660 and number clicks = 28325,

##### Number of cookies

```
SD = sqrt(0.5 x 0.5 / (345543 + 344660)) = 0.000602
margin of error = SD x 1.96 = 0.00118
cinterval = (0.5 - 0.00118, 0.5 + 0.00118) = (0.49882, 0.50118)
p = 345543 / (345543 + 344660) = 0.50064
```

It passes the sanity check because the number is within the confidence
interval.

##### **Number of clicks:**

```
SD = sqrt(0.5 x 0.5 / (28378 + 28325)) = 0.0021
margin of error = SD x 1.96 = 0.004116
cinterval = (0.5 - 0.004116, 0.5 + 0.004116) = (0.495884, 0.504116)
p = 28378 / (28378 + 28325) = 0.500467
```

It passes the sanity check because the number is within the confidence
interval.

##### **Click-through-probability:**

As this metric equals the number of clicks divided by the number of cookies,
this metric also passes the sanity check.

### Follow-Up Experiment: How to Reduce Early Cancellations
