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

### Splitting the dataset

* Function *split_data_set()*

    ```Python
    def split_data_set(data_set, axis, value):
        ret_data_set = []
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                reduced_feat_vec = feat_vec[:axis]
                reduced_feat_vec.extend(feat_vec[axis+1:])
                ret_data_set.append(reduced_feat_vec)
        return ret_data_set
    ```

* Function *choose_best_feature_to_split(data_set)*

    ```Python
    def choose_best_feature_to_split(data_set):
        num_features = len(data_set[0]) - 1
        base_entropy = calc_shannon_ent(data_set)
        best_info_gain = 0.0
        best_feature = -1
        for i in range(num_features):
            feat_list = [example[i] for example in data_set]
            unique_vals = set(feat_list)
            new_entropy = 0.0
            for value in unique_vals:
                sub_data_set = split_data_set(data_set, i, value)
                prob = len(sub_data_set)/float(len(data_set))
                new_entorpy += prob * calc_shannon_ent(sub_data_set)
            info_gain = base_entropy - new_entropy
            if (info_gain > best_info_gain):
                best_info_gain = info_gain
                best_feature = I
        return best_feature
    ```

### Recursively building the tree

If our dataset has run out of attributes but the class labels are not all the same, we must decide what to call that leaf node. In this situation, we'll take a majority vote.

* Function *majority_cnt()*

    ```Python
    import operator
    def majority_cnt(class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
        return sorted_class_count[0][0]
    ```

* Function *create_tree()*

    ```Python
    def createTree(data_set, labels):
        class_list = [example[-1] for example in data_set]
        # stop when all classes are equal
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        # when no more features, return majority
        if len(data_set[0]) == 1:
            return majority_cnt(class_list)
        best_feat = choose_best_feature_to_split(data_set)
        best_feat_label = labels[best_feat]
        my_tree = {best_feat_label:{}}
        del(labels[best_feat])
        feat_values = [example[best_feat] for example in data_set]
        unique_vals = set(feat_values)
        for value in unique_vals:
            sub_labels = labels[:]
            my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
        return my_tree
    ```

## Plotting trees in Python with Matplotlib annotations

* Plot tree nodes with text annotations

    ```Python
    import matplotlib.pyplot as plt

    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_notde = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arraystyle="<-")

    def plot_node(node_txt, center_pt, parent_pt, node_type):
        create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt, bbox=node_type, arrowprops=arrow_args)

    def create_plot():
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        create_plot.ax1 = plt.subplot(111, frameon=False)
        plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
        plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
        plt.show()
    ```

> I need to modify the codes to apply this function to Python 3.x. For now, I'll just skip this section.

## Testing and storing the classifier

### Test: using the tree for classification

* Function *classify()*

    ```Python
    def classify(input_tree, feat_labels, test_vec):
        # convert to list by hand ???
        first_str = list(input_tree.keys())[0]
        second_dict = input_tree[first_str]
        feat_index = feat_labels.index(first_str)
        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = classify(second_dict[key], feat_labels, test_vec)
                else:
                    class_label = second_dict[key]
        return class_label
    ```

> When classifying the data, the 'labels' variable is changed by the create_tree() function. We need to retrieve the labels again from create_data().

### Use: persisting the decision tree

Building the tree would take a long time when it comes to large datasets. It would be a waste of time to build the tree every time. We're going to use the Python module, **pickle**, to serialize objects. Serializing objects allows you to store them for later use.

```Python
def store_tree(input_tree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()

def grab_tree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
```

In this way, we can **distill** the dataset into some knowledge, and we use that knowledge only when we want to classify something.

## Example: using decision trees to predict contact lens type

The **Lenses dataset** is a number of observations based on patients' eye conditions and the type of contact lenses the doctor prescribed. The classes are hard, soft, and no contact lenses. The data is from the UCI database repository and is modified slightly so that it can be displayed easier.

To load the data:

```Python
>>> fr = open('lenses.txt')
>>> lenses = [inst.strip().split('\t') for inst in fr.readlines()]
>>> lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
>>> lenses_tree = trees.create_tree(lenses, lenses_labels)
```

However, our tree matches our data too well. This problem is known as **overfitting**. To reduce the problem of overfitting, we can prune the tree. This will go through and remove some leaves. If a leaf node adds only a little information, it will be cut off and merged with another leaf. We will investigate this further when we revisit decision tree in chapter 9.

In chapter 9 we'll also investigate another decision tree algorithm called CART. The algorithm we used in this chapter, ID3, is good but not the best. ID3 can't handle numeric values. We could use continuous values by quantizing them inot discrete bins, but ID3 sufferes from other problems if we have too many splits.

## Summary

There are other decision tree-generating algorithms. The most popular are C4.5 and CART. CART will be addressed in chapter 9 when we use it for regression.
