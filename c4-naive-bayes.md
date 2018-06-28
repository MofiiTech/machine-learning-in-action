---
Type: Reading Notes
Book Title: Machine Learning In Action
Subtitle: Chapter 4 - Classifying with probability theory: naive Bayes
---

# Chapter 4 - Classifying with probability theory: naive Bayes

## Classifying with Bayesian decision thoery

> Naive Bayes
> Pros: Works with a small amount of data, handles multiple classes
> Cons: Sensitive to how the input data is prepared
> Works with: Nominal values

**Bayesian decision thoery**: choosing the decision with the highest probability.

## Conditional probability

**Bayes' rule**: *p(c/x) = p(x/c) x p(c) / p(x)*

## Classifying with conditional probabilities

With these definitions, we can define the Bayesian classification rule as:

  - If P(c<sub>1</sub> \| x, y) > P(c<sub>2</sub> \| x, y), the class is c<sub>1</sub>.
  - If P(c<sub>1</sub> \| x, y) < P(c<sub>2</sub> \| x, y), the class is c<sub>2</sub>.
