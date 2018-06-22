---
Type: Reading Notes
Book Title: Machine Learning In Action
Subtitle: Chapter 3 - Splitting datasets one feature at a time: decision trees
---

# Chapter 2 - Splitting datasets one feature at a time: decision trees

## Tree construction

>Decision trees
>Pros: Computationally cheap to use, easy for humans to understand learned results, missing values OK, can deal with irrelevant features
>Cons: Prone to overfitting
>Works with: Numeric values, nominal values

Before we write the function *createBranch()* in Python, we need to split the dataset. If we split on an attribute and it has four possible values, then we'll split the data four ways and create four separate branches. We'll follow the [ID3 algorithm](https://en.wikipedia.org/wiki/ID3_algorithm), which tells us how to split the data and when to stop splitting it.

### Information gain

We choose to split our dataset in a way that makes our unorganized data more organized. One way to do this is to measure the information. Using information theory, you can measure the information before and after the split.

The change in the information before and after the split is known as the **information gain**. We can split the dataset across every feature to see which split gives the highest information gain. The split with the highest information gain is the best option. The measure of information of a set is known as the **Shannon entropy**, or just **entroppy**.

Entropy is defined as the expected value of the information. If you're classifying something that can take on multiple values, the information for symbol x<sup>i</sup> is defined as

$l(x_i) = \log_2 p(x_i)$
