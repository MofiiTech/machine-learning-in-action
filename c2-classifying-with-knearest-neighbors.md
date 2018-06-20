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

### Example: Classifying movies into romance or action movies

* Given six movies with the number of kisses and kicks in each movie.

|Movie title|# of kicks|# of kisses|Type of movie|
|-|-|-|-|
|California Man|3|104|Romance|
|He's Not Really into Dudes|2|100|Romance|
|Beautiful Woman|1|81|Romance|
|Kevin Longblade|101|10|Action|
|Robo Slayer 3000|99|5|Action|
|Amped II|98|2|Action|
|?|18|90|Unknown|
