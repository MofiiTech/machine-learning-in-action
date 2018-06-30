---
Type: Reading Notes
Book Title: Machine Learning In Action
Subtitle: Chapter 4 - Classifying with probability theory: naive Bayes
---

# Chapter 4 - Classifying with probability theory: naive Bayes

## Classifying with Bayesian decision theory

> Naive Bayes
>
> Pros: Works with a small amount of data, handles multiple classes
>
> Cons: Sensitive to how the input data is prepared
>
> Works with: Nominal values

**Bayesian decision thoery**: choosing the decision with the highest probability.

## Conditional probability

**Bayes' rule**: *p(c/x) = p(x/c) x p(c) / p(x)*

## Classifying with conditional probabilities

With these definitions, we can define the Bayesian classification rule as:

  - If P(c<sub>1</sub> \| x, y) > P(c<sub>2</sub> \| x, y), the class is c<sub>1</sub>.
  - If P(c<sub>1</sub> \| x, y) < P(c<sub>2</sub> \| x, y), the class is c<sub>2</sub>.

## Document classification with naive Bayes

Let's assume that our vocabulary is 1,000 words long. In order to generate distributions, we need enough data examples. Statistics tells us that if we need N samples for one feature, we need N<sup>10</sup> for 10 features, and N<sup>1000</sup> for 1,000 features.

Here we assume independence among the features, which means one feature or word is just as likely by itself as it is next to other words. This is what is meant by **naive** in the **naive Bayes classifier**. The other assumption we make is that every feature is equally important. Despite the mirror flaws of these assumptions, naive Bayes works well in practice.

## Classifying text with Python

To see this in action, let's make a quick filter for an online message board that flags a message as inappropriate if the author uses negative or abusive language. We'll have two categories: abusive (1) and not (0).

### Prepare: making word vectors from text

* Function *load_data_set()*

    ```py
    def load_data_set():
        posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        class_vec = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not
        return posting_list, class_vec
    ```

* Function *create_vocab_list()*

    ```py
    def create_vocab_list(data_set):
        vocab_set = set([])
        for document in data_set:
            vocab_set = vocab_set | set(document) # union of two sets
        return list(vocab_set)
    ```

* Function *set_of_words_to_vec()*

    ```py
    def set_of_words_to_vec(vocab_list, input_set):
        return_vec = [0] * len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                return_vec[vocab_list.index(word)] = 1
            else:
                print("the word: {:s} is not in my Vocabulary".format(word))
        return return_vec
    ```

### Train: calculating probabilities from word vectors

Now we are going to calculate the probabilities with theses numbers: *p(ci|w) = p(w|ci) x p(ci) / p(w)*.

  - p(ci\|w): probability of a post is abusive given its vector
  - p(w\|ci): probability of this post among all abusive posts
  - p(ci): probability of abusive posts
  - p(w): probability of this post

* Function *train_nb0()*

    ```py
    from numpy import *
    def train_nb0(train_matrix, train_category):
        num_train_docs = len(train_matrix)
        num_words = len(train_matrix[0])
        p_abusive = sum(train_category) / num_train_docs
        p0num = zeros(num_words)
        p1num = zeros(num_words)
        p0denom = 0.0
        p1denom = 0.0
        for i in range(num_train_docs):
            if train_category[i] == 1:
                p1num += train_matrix[i]
                p1denom += sum(train_matrix[i])
            else:
                p0num += train_matrix[i]
                p0denom += sum(train_matrix[i])
        p1vect = p1num / p1denom
        p0vect = p0num / p0denom
        return p0vect, p1vect, p_abusive
    ```

Train the model:

```py
>>> import bayes
>>> list_posts, list_classes = bayes.load_data_set()
>>> vocab_list = bayes.create_vocab_list(list_posts)
>>> train_mat = []
>>> for post in list_posts:
...   train_mat.append(bayes.set_of_words_to_vec(vocab_list, post))
...
>>> p0v, p1v, p_ab = bayes.train_nb0(train_mat, list_classes)
```

### Test: modifying the classifier for real-world conditions

When we attempt to classify a document, we multiply a lot of probabilities together to get the probability that a document belongs to a given class. This will look something like *p(w0|1) p(w1|1) p(w2|1)*. If any of these numbers are 0, then when we multiply them together we get 0. To lessen the impact of this, we'll initialize all of our occurence counts to 1, and we'll initialize the denominators to 2.

Now we modify the function *train_nb0()* in *bayes.py*:

```py
p0num = ones(num_words)
p1num = ones(num_words)
p0denom = 2.0
p1denom = 2.0
```

Another problem is underflow: doing too many multiplications of small numbers. One solution to this is to take the natural logarithm of this product.

To modify our classifier to account for this:

```py
p1vect = log(p1num/p1denom)
p0vect = log(p0num/p0denom)
```

Now we can build the classifier.

* Function *classify_nb()*:

    ```py
    def classify_nb(vec2classify, p0vec, p1vec, p_class1):
        p1 = sum(vec2classify * p1vec) + log(p_class1)
        p0 = sum(vec2classify * p0vec) + log(1.0 - p_class1)
        if p1 > p0:
            return 1
        else:
            return 0
    ```

* Function *testing_nb()*

    ```py
    def testing_nb():
        list_posts, list_classes = load_data_set()
        vocab_list = create_vocab_list(list_posts)
        train_mat = []
        for post in list_posts:
            train_mat.append(set_of_words_to_vec(vocab_list, post))
        p0v, p1v, p_ab = train_nb0(array(train_mat), array(list_classes))
        test_entry = ['love', 'my', 'dalmation']
        this_doc = array(set_of_words_to_vec(vocab_list, test_entry))
        print()
    ```

* Sample output

    ```
    >>> bayes.testing_nb()
    ['love', 'my', 'dalmation'] classified as: 0
    ['stupid', 'garbage'] classified as: 1
    ```

### Prepare: the bag-of-words document model

Up until this point we've treated the presence or absence of a word as a feature. This could be described as a set-of-words model. If a word appears more than once in a document, that might convey some sort of information about the document over just the word occurring in the document or not. This approach is known as a **bag-of-words** model.

* Function *bag_of_words_to_vec_mn()*

    ```py
    def bag_of_words_to_vec_mn(vocab_list, input_set):
        return_vec = [0] * len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                return_vec[vocab_list.index(word)] += 1
        return return_vec
    ```

## Example: classifying spam email with naive Bayes

### Prepare: tokenizing text

If we have a text string, we can split it using the Python *string.split()* method.

```py
>>> my_sent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
>>> my_sent.split()
['This', 'book', 'is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M.L.', 'I', 'have', 'ever', 'laid', 'eyes', 'upon.']
```

Further we can use regular expressions to split up the sentence on anything that isn't a word or number.

```py
>>> import re
>>> regex = re.compile('\\W*')
>>> list_tokens = regex.split(my_sent)
>>> list_tokens
['This', 'book', 'is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M', 'L', 'I', 'have', 'ever', 'laid', 'eyes', 'upon', '']
```

Now remove empty strings and change capitalized words to lowercase.

```py
>>> [tok.lower() for tok in list_tokens if len(tok > 0)]
```

### Test: cross validation with naive Bayes

* Function *text_parse()*

    ```py
    def text_parse(big_string):
        import re
        list_tokens = re.split(r'\W*', big_string)
        return [tok.lower() for tok in list_tokens if len(tok) > 2]
    ```

* Function *spam_test()*

    ```py
    def spam_test():
        doc_list = []
        class_list = []
        full_text = []
        for i in range(1, 26):
            word_list = text_parse(open('email/spam/{:d}.txt'.format(i)).read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)
            word_list = text_parse(open('email/ham/{:d}.txt'.format(i)).read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        vocab_list = create_vocab_list(doc_list)
        training_set = range(50)
        test_set = []
        for i in range(10):
            rand_index = int(random.uniform(0, len(training_set)))
            test_set.append(training_set[rand_index])
            del(training_set[rand_index])
        train_mat = []
        train_classes = []
        for doc_index in training_set:
            train_mat.append(set_of_words_to_vec(vocab_list, doc_list[doc_index]))
            train_classes.append(class_list[doc_index])
        p0v, p1v, p_spam = train_nb0(array(train_mat), array(train_classes))
        error_count = 0
        for doc_index in test_set:
            word_vec = set_of_words_to_vec(vocab_list, doc_list[doc_index])
            if classify_nb(array(word_vec), p0v, p1v, p_spam) != class_list[doc_index]:
                error_count += 1
        print('The error rate is: {}'.format(error_count/len(test_set)))
    ```

> When classifying the provided email texts, an error is returned: ```UnicodeDecodeError: 'utf-8' codec can't decode byte 0xae in position 199: invalid start byte```. The problem is that some characters cannot be converted to utf-8 format. They should be replaced with utf-8 characters or removed.

* Sample output

    ```
    >>> bayes.spam_test()
    The error rate is: 0.0
    >>> bayes.spam_test()
    The error rate is: 0.1
    ```

Since we randomly pick 10 emails as our test set, the outcome is different each time. Overall, the error rate is around 6%.

## Example: using naive Bayes to reveal local attitudes from personal ads

In this example, we'll take some data from personal ads from multiple people for two different cities in the United States. We're going to see if people in different cities use different words.

### Collect: importing RSS feeds

We are going to use [Universal Feed Parser](https://github.com/kurtmckee/feedparser) to read texts in RSS form.

We're going to use the personal ads from Craigslist, and hopefully we'll stay Terms Of Service complaint. To open the RSS feed from Craigslist:

```py
>>> import feedparser
>>> ny = feedparser.parse('http://newyork.carigslist.org/stp/index.rss')
```

> Update: Carigslist servers seem to be down. I'll keep this section on hold for now.
