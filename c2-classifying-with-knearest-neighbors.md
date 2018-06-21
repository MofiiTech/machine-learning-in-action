---
Book Title: Machine Learning In Action
Subtitle: Chapter 2 - Classifying with k-Nearest Neighbors
---

# Chapter 2 - Classifying with k-Nearest Neighbors

## Classifying with distance measurements

> k-Nearest Neighbors
> - Pros: High accuracy, insensitive to outliers, no assumptions about data
> - Cons: Computationally expensive, requires a lot of memory
> - Works with: Numeric values, nominal values

* The first machine-learning algorithm is k-Nearest Neighbors (kNN). When given a new piece of data, we compare the new piece of data with our training set. We look at the k most similar pieces of data and take a majority vote from the k pieces of data, and the majority is the new class we assign to the data we were asked to classify.

### Prepare: importing data with Python

* Create a Python module: *kNN.py*

    ```Python
    from numpy import *
    import operator

    def createDataSet():
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
    return group, labels
    ```

### Putting the kNN classification algorithm into action

* Function *classify0()*

    ```Python
    def classify0(inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis = 1)
        distances = sqDistances ** 0.5
        classCount = {}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
    ```
