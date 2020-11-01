import nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import brown
from tqdm import tqdm


# datastructure for storing the id and val mapping
class unique_dict:
    def __init__(self):
        self.id = 0
        self.id2val = dict()
        self.val2id = dict()
        self.vals = set()

    def get_length(self):
        return self.id

    def store(self, val):
        self.id2val[self.id] = val
        self.val2id[val] = self.id
        self.vals.add(val)
        self.id += 1

    def fetch(self, key, by='id'):
        if by == 'id':
            return self.id2val[key]
        elif key in self.vals:
            return self.val2id[key]
        else:
            return None


'''
    training functions
'''

def generate_unique_words(train_corpus):
    words = unique_dict()

    for tagged_sent in train_corpus:
        for (word, tag) in tagged_sent:
            if words.fetch(word.lower(), 'val') == None:
                words.store(word.lower())

    return words

def generate_unique_tags():
    tagset = set([tag for [word, tag] in brown.tagged_words(tagset='universal')])
    tags = unique_dict()

    for tag in tagset:
        if tags.fetch(tag, 'val') == None:
            tags.store(tag)

    return tags

'''
    Generates three useful matrices
    transition matrix for the tags
    emission matrix for the tag and word
    probability of occurence vector for tags
'''
def generate_useful_matrices(train_corpus, words, tags, unknown_prob= 0.0000001):
    cond_prob_tags = np.zeros([tags.id, tags.id])
    cond_prob_tag_word = np.zeros([tags.id, words.id])
    prob_tags = np.zeros([tags.id])

    tic = time.time()
    for tagged_sent in train_corpus:
        for index in range(len(tagged_sent)):
            word = tagged_sent[index][0]
            tag = tagged_sent[index][1]

            word_id = words.fetch(word.lower(), 'val')
            tag_id = tags.fetch(tag, 'val')

            prob_tags[tag_id] += 1
            cond_prob_tag_word[tag_id, word_id] += 1
            if index != len(tagged_sent) - 1:
                next_tag_id = tags.fetch(tagged_sent[index + 1][1], 'val')
                cond_prob_tags[tag_id, next_tag_id] += 1
    toc = time.time()
    print('Time elapsed for frequency computation:', toc - tic)

    tic = time.time()
    cond_prob_tags = np.divide(cond_prob_tags, np.reshape(prob_tags, [-1, 1]) + np.reshape(prob_tags == 0, [-1, 1]))
    cond_prob_tag_word = np.divide(cond_prob_tag_word, np.reshape(prob_tags, [-1, 1]) + np.reshape(prob_tags == 0, [-1, 1]))
    prob_tags = np.divide(prob_tags, np.sum(prob_tags))
    toc = time.time()
    print('Time elapsed for normalization:', toc - tic)

    # handling zero prob cases
    cond_prob_tags[cond_prob_tags == 0] = unknown_prob
    cond_prob_tag_word[cond_prob_tag_word == 0] = unknown_prob
    prob_tags[prob_tags == 0] = unknown_prob

    return cond_prob_tags, cond_prob_tag_word, prob_tags


'''
    Testing utils
'''

# vitterbi algorithm
def assign_pos(token_list, cond_prob_tags, cond_prob_tag_word, prob_tags, words, tags, unknown_prob=0.0000001):
    if len(token_list) == 0:
        return []

    score_matrix = np.zeros([tags.id, len(token_list)])
    back_pointer = np.zeros([tags.id, len(token_list)])

    # intialization
    word_id = words.fetch(token_list[0].lower(), 'val')
    for i in range(tags.id):
        if word_id == None:
            score_matrix[i, 0] = prob_tags[i] * unknown_prob
        else:
            score_matrix[i, 0] = prob_tags[i] * cond_prob_tag_word[i, word_id]
        back_pointer[i, 0] = -1;

    # Iteration Step
    for t in range(len(token_list)):
        if t != 0:
            for i in range(tags.id):
                word_id = words.fetch(token_list[t].lower(), 'val')
                total_transition_vector = np.multiply(score_matrix[:, t - 1], cond_prob_tags[:, i])
                arg_tag_max = np.argmax(total_transition_vector)
                back_pointer[i, t] = arg_tag_max
                if word_id == None:
                    score_matrix[i, t] = total_transition_vector[arg_tag_max] * unknown_prob
                else:
                    score_matrix[i, t] = total_transition_vector[arg_tag_max] * cond_prob_tag_word[i, word_id]

    # sequence identification
    tag_index = np.zeros([len(token_list)])
    tag_index[-1] = np.argmax(score_matrix[:, len(token_list) - 1])
    for i in reversed(range(len(token_list) - 1)):
        tag_index[i] = back_pointer[int(tag_index[i + 1]), int(i + 1)]
    return [tags.fetch(index, 'id') for index in tag_index]

# function for analysing testing data
def analyze_test_data(test_corpus, cond_prob_tags, cond_prob_tag_word, prob_tags, words, tags, unknown_prob=0.000001):
    confusion_matrix = np.zeros([tags.id, tags.id], dtype=np.int32)

    for test_sent_tagged in tqdm(test_corpus):
        test_tag_list = [item[1] for item in test_sent_tagged]
        test_token_list = [item[0] for item in test_sent_tagged]

        predicted_tag_list = assign_pos(test_token_list, cond_prob_tags, cond_prob_tag_word, prob_tags, words, tags, unknown_prob)
        for (predicted_tag, test_tag) in zip(predicted_tag_list, test_tag_list):
            confusion_matrix[tags.fetch(predicted_tag, 'val'), tags.fetch(test_tag, 'val')] += 1

    return confusion_matrix

# function which does cross validation analysis by generating confusion matrix
def hmm_confusion(unknown_prob=0.0000001):
    tags = generate_unique_tags()
    confusion_matrix_train = np.zeros([tags.id, tags.id], dtype=np.int32)
    confusion_matrix_test = np.zeros([tags.id, tags.id], dtype=np.int32)
    tag_histogram = np.zeros([tags.id], dtype=np.float32)
    corpus = np.array(brown.tagged_sents(tagset='universal'))
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(corpus)

    for train_index, test_index in kf.split(corpus):
        train_corpus = corpus[train_index]
        test_corpus = corpus[test_index]
        print('training size:', len(train_index))
        print('testing size:', len(test_index))
        words = generate_unique_words(train_corpus)
        cond_prob_tags, cond_prob_tag_word, prob_tags = generate_useful_matrices(train_corpus, words, tags, unknown_prob)
        tag_histogram += prob_tags
        confusion_matrix_test += analyze_test_data(test_corpus, cond_prob_tags, cond_prob_tag_word, prob_tags, words, tags, unknown_prob)
        confusion_matrix_train += analyze_test_data(train_corpus, cond_prob_tags, cond_prob_tag_word, prob_tags, words, tags, unknown_prob)

    tag_histogram /= np.sum(tag_histogram)
    return confusion_matrix_train, confusion_matrix_test, tag_histogram

# to analyze the confusion matrix (to be tested)
def analyze_confusion(confusion_matrix, tag_histogram, tags, data='test'):

    # accuracy computation
    total_examples = np.sum(confusion_matrix)
    correct_predictions = np.trace(confusion_matrix)
    print('The overall accuracy of the hmm model is:', correct_predictions * 100 / total_examples)

    # storing the confusion matrix in the form of csv file
    tag_list = [tags.fetch(i, 'id') for i in range(tags.id)]
    confusion_df = pd.DataFrame(confusion_matrix, index=tag_list, columns=tag_list)
    confusion_df.to_csv('hmm_confusion_matrix_' + data + '.csv')

    # plotting the heat map
    plt.figure(figsize = (20, 20))
    confusion_figure = sns.heatmap(confusion_matrix, annot=True, xticklabels=tag_list, yticklabels=tag_list)
    plt.savefig('hmm_confusion_figure_' + data + '.png')

    # per POS tag statistics
    per_pos_dict = {'tag': [], 'precision': [], 'recall': [], 'f1-score': []}
    for tag_id in range(tags.id):
        per_pos_dict['precision'].append(confusion_matrix[tag_id, tag_id] / np.sum(confusion_matrix[tag_id, :]))
        per_pos_dict['recall'].append(confusion_matrix[tag_id, tag_id] / np.sum(confusion_matrix[:, tag_id]))
        per_pos_dict['tag'].append(tags.fetch(tag_id, 'id'))
        per_pos_dict['f1-score'].append(2 * per_pos_dict['precision'][tag_id] * per_pos_dict['recall'][tag_id] / (per_pos_dict['recall'][tag_id] + per_pos_dict['precision'][tag_id]))
    per_pos_df = pd.DataFrame(per_pos_dict)
    per_pos_df.to_csv('hmm_per_pos_accuracy_' + data + '.csv')

    # scatter plot for frequency vs f1-score for every tag
    tag_f1_score = np.nan_to_num(per_pos_dict['f1-score'])
    plt.figure()
    plt.scatter(tag_histogram, tag_f1_score)
    plt.title('Relative Frequency vs f1-score scatter plot')
    plt.xlabel('Relative Frequency of tag')
    plt.ylabel('F1 score')
    plt.savefig('hmm_scatter_plot_' + data + '.png')


if __name__ == '__main__':

    # reproducing all the statistics in the report
    confusion_matrix_train, confusion_matrix_test, tag_histogram = hmm_confusion()
    analyze_confusion(confusion_matrix_test, tag_histogram, tags, data='test')
    analyze_confusion(confusion_matrix_train, tag_histogram, tags, data='train')
    
    # unknown probability hyperparameter optimization
    unknown_prob_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    test_accuracy = []
    train_accuracy = []

    # computation of confusion matrix for accuracy
    for unknown_prob in unknown_prob_list:
        confusion_matrix_train, confusion_matrix_test, _ = hmm_confusion(unknown_prob)
        test_accuracy.append(100 * np.trace(confusion_matrix_test) / np.sum(confusion_matrix_test))
        train_accuracy.append(100 * np.trace(confusion_matrix_train) / np.sum(confusion_matrix_train))

    # plotting the graph
    plt.figure(1)
    plt.title('Zero order back off hyperparameter optimization')
    plt.xlabel('Probability Value')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.plot(unknown_prob_list, train_accuracy, label='train_accuracy')
    plt.plot(unknown_prob_list, test_accuracy, label='test_accuracy')
    plt.show
    plt.savefig('zero_order_back_off_optimization.png')
