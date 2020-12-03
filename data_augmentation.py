import random
from random import shuffle
import glob
import json
import os
import shutil
import time
random.seed(1)

#stop words list
filename = "stopword.txt"

def read_stop_words(filename):
    data = []
    with open(filename, 'r') as f:
        while(True):
            line = f.readline()
            if line=='' or line =='\n':
                break
            data.append(line)
        return data

stop_words = read_stop_words(filename)

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


"""
replace the word with Synonym from wordnet in nltk
"""

from nltk.corpus import wordnet

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

"""
Random delete the word with probability p
"""

def random_deletion(words, p):

    #if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

"""
Random swap two words n times
"""

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

"""
Random insert n words
"""
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


"""
the main function
"""


def augmentation(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    #synonym_replacement
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    #random_insertion
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    #random_swap
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    #random_deletion
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    #number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    #append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences


start = time.time()

# Variables
in_dir = "RACE" # the in_dir
out_dir = "G_Data" # the out_dir
num_aug = 2 # number of augmented files to generate per randomly chosen file
alpha = 0.1 # how much to change each sentence
number = 2000 # number number of files to be augmented (to be added to dataset)

# read each file, copy num_aug times, and replace article with augment
def augment(path):
    filenames = random.sample(glob.glob(path + "/*txt"), number)
    for filename in filenames:
        new_filename = filename[:-4] + "_" + str(0) + filename[-4:]
        os.rename(filename, new_filename)
        for i in range(1, num_aug + 1):
            copy_filename = new_filename[:-5] + str(i) + new_filename[-4:]
            shutil.copyfile(new_filename, copy_filename)
        with open(new_filename, 'r', encoding='utf-8') as fpr:        
            data_raw = json.load(fpr)
            article = data_raw['article']
        aug_article = augmentation(article, alpha_sr=alpha, alpha_ri=alpha, 
                          alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for i in range(num_aug+1):
            write_file = new_filename[:-5] + str(i) + new_filename[-4:]
            with open(write_file, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                data_raw['article'] = aug_article[i]
            with open(write_file, 'w', encoding='utf-8') as fpr:
                fpr.write(json.dumps(data_raw))


# copy the entire in_dir directory
shutil.copytree(in_dir, out_dir)

# run data augmentation
train_path_h = out_dir + "/train/high"
augment(train_path_h)
train_path_m = out_dir + "/train/middle"
augment(train_path_m)        

shutil.make_archive(out_dir, 'zip', out_dir)

end = time.time()
print("Computation time:", end - start)