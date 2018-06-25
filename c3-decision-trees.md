---
Type: Reading Notes
Book Title: Machine Learning In Action
Subtitle: Chapter 3 - Splitting datasets one feature at a time: decision trees
---

# Chapter 3 - Splitting datasets one feature at a time: decision trees

## Tree construction

>Decision trees
>Pros: Computationally cheap to use, easy for humans to understand learned results, missing values OK, can deal with irrelevant features
>Cons: Prone to overfitting
>Works with: Numeric values, nominal values

Before we write the function *createBranch()* in Python, we need to split the dataset. If we split on an attribute and it has four possible values, then we'll split the data four ways and create four separate branches. We'll follow the [ID3 algorithm](https://en.wikipedia.org/wiki/ID3_algorithm), which tells us how to split the data and when to stop splitting it.

### Information gain

We choose to split our dataset in a way that makes our unorganized data more organized. One way to do this is to measure the information. Using information theory, you can measure the information before and after the split.

The change in the information before and after the split is known as the **information gain**. We can split the dataset across every feature to see which split gives the highest information gain. The split with the highest information gain is the best option. The measure of information of a set is known as the **Shannon entropy**, or just **entroppy**.

Entropy is defined as the expected value of the information. If you're classifying something that can take on multiple values, the information for symbol **x<sup>i</sup>** is defined as **l(x<sub>i</sub>) = log<sub>2</sub>p(x<sub>i</sub>)**, where **p(x<sub>i</sub>)** is the probability of choosing this class.

When calculating entropy, you need the expected value of all the information of all possible values of our class. This is given by **H = - sum(p(x<sub>i</sub>)log<sub>2</sub>p(x<sub>i</sub>))**.

* Function *calc_shannon_ent()*:

    ```Python
    def calc_shannon_ent(data_set):
        num_entries = len(data_set)
        label_counts = {}
        for feat_vec in data_set:
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
        shannon_ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key])/num_entries
            shannon_ent -= prob * log(prob, 2)
        return shannon_ent
    ```

The higher the entropy is, the more mixed up the data is. We will split the dataset in a way that will give us largest information gain.

> Another common measure of disorder in a set is the **Gini impurity**, which is the probability of choosing an item from the set and the probability of that item being misclassified.
