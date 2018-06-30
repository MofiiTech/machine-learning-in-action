from numpy import *


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document) # union of two sets
    return list(vocab_set)


def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: {:s} is not in my Vocabulary".format(word))
    return return_vec


def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / num_train_docs
    p0num = ones(num_words)
    p1num = ones(num_words)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1num += train_matrix[i]
            p1denom += sum(train_matrix[i])
        else:
            p0num += train_matrix[i]
            p0denom += sum(train_matrix[i])
    p1vect = log(p1num / p1denom)
    p0vect = log(p0num / p0denom)
    return p0vect, p1vect, p_abusive


def classify_nb(vec2classify, p0vec, p1vec, p_class1):
    p1 = sum(vec2classify * p1vec) + log(p_class1)
    p0 = sum(vec2classify * p0vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    list_posts, list_classes = load_data_set()
    vocab_list = create_vocab_list(list_posts)
    train_mat = []
    for post in list_posts:
        train_mat.append(set_of_words_to_vec(vocab_list, post))
    p0v, p1v, p_ab = train_nb0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words_to_vec(vocab_list, test_entry))
    print(test_entry, end='')
    print(' classified as: ', end='')
    print(classify_nb(this_doc, p0v, p1v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words_to_vec(vocab_list, test_entry))
    print(test_entry, end='')
    print(' classified as: ', end='')
    print(classify_nb(this_doc, p0v, p1v, p_ab))


def text_parse(big_string):
    import re
    list_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_tokens if len(tok) > 2]


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
    training_set = list(range(50))
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
