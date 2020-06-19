import sys

import numpy as np
import time
import random
import operator
import copy
from trie import TrieDict


def ag(t, trie, points, multiset, population, pop_size=200, parent_pop_size=40, mutation_chance=0.8, crossover_chance=0.4):
    time_to_end = time.time() + t
    population += random_population(trie, multiset, pop_size - len(population))
    evaluated = evaluate_population(trie, population, points)
    best_word, best_value = evaluated[0]

    while time.time() < time_to_end:
        # select parents
        parents = select_parents(evaluated, parent_pop_size)
        children = []
        while len(children) < pop_size - parent_pop_size:
            ch1 = random.choice(parents)
            ch2 = random.choice(parents)
            if random.random() < crossover_chance:
                ch1, ch2 = crossover(ch1, ch2)
            if random.random() < mutation_chance:
                ch1 = mutate_word(ch1, trie, multiset)
                ch2 = mutate_word(ch2, trie, multiset)
            children.append(ch1)
            children.append(ch2)

        population = parents + children

        evaluated = evaluate_population(trie, population, points)
        if evaluated[0][1] > best_value:
            best_word, best_value = evaluated[0]

    return best_word, best_value


def random_population(trie, multiset, n):
    return [trie.get_random_possible_from_multiset(multiset) for i in range(n)]


def select_parents(evaluated, n):
    ''' wybór rodziców '''
    ev = copy.copy(evaluated)

    # half elitism
    res = [ev[0] for ev in evaluated[:n // 2]]

    # half linear ranking
    ev = ev[n//2:]
    ev_words = [w[0] for w in ev]
    ev_values = np.array([w[1] for w in ev])
    res2 = list(np.random.choice(ev_words, size=n-len(res), replace=False, p=ev_values / sum(ev_values)))

    return res + res2


def evaluate_population(trie, population, points):
    ''' fitness całej populacji '''
    val_pop = {word: value(word, points, multiset, trie) for word in population}
    sorted_val_pop = sorted(val_pop.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_val_pop


def crossover(par1, par2, rate=0):
    ''' mix two parents aaaaaaa,bbbbbbb into aaaabbbb, bbbbaaaa'''
    point = max(1, random.randint(1, len(par1)))
    return par1[:point] + par2[point:], par2[:point] + par1[point:]


def value(word, points, multiset, trie):
    ''' fitness '''
    score = 0
    multiset = copy.copy(multiset)
    correct = True
    for letter in word:
        letter = letter.lower()
        if letter in multiset:
            score += points[letter]
            multiset.remove(letter)
        else:
            correct = False
    if trie.contains(word) and correct:
        return score
    else:
        # return int(sqrt(score))
        return int(score / len(word))


def multiset_check(word, multiset):
    """ sprawdza czy słowo należy do multisetu"""
    mset = copy.copy(multiset)
    if word:
        for letter in word:
            if letter in mset:
                mset.remove(letter)
            else:
                return False
    return True


def mutate_word(word, trie, multiset):
    ''' usuwa losowo r ostatnich liter,
    a pozostałe wykorzystuje do szukanie prefixu '''
    mset = copy.copy(multiset)
    prefix = word

    # dopoki nie da sie złozyć z liter z multisetu to usun ostatnia litere
    while not multiset_check(prefix, mset):
        prefix = prefix[:-1]
    for letter in prefix:
        mset.remove(letter)

    if random.random() < 0.5:
        word = add_random_letters(prefix, multiset)

    if random.random() < 0.5:
        word = swap_letter(prefix, mset)

    return word


def add_random_letters(word, multiset):
    ''' dodaje losową ilosc liter z multisetu'''
    mset = copy.copy(multiset)

    n = len(mset)
    # p=[1/2,1/4,1/8...1/x,1/x]
    weights = [1 / 2 ** i for i in range(1, n - 1)]
    weights.append(1 - (sum(weights)))

    # ile liter dodać
    num_of_letter_to_add = random.choices(list(range(1, n)), weights)
    # litery do dodania
    letters_to_add = np.random.choice(mset, size=num_of_letter_to_add, replace=False)
    word = insert_letters_into_word(''.join(letters_to_add), word, random.randint(0, len(word)))
    return word


def swap_letter(word, multiset):
    ''' zamienia jedna litere w słowie na wybrane z dostepnych z multisetu'''
    if multiset:
        random_letter = random.choice(multiset)
        index = random.randint(0, len(word) - 1)
        wordlist = list(word)
        wordlist[index] = random_letter
        word = ''.join(wordlist)
    return word


def insert_letters_into_word(letters, word, pos):
    ''' wstawia litery do środka wyrazu'''
    return word[:pos] + letters + word[pos:]


if __name__ == '__main__':
    first_line = input()
    t, n, s = [int(num) for num in first_line.split()]
    points = {}
    multiset = []
    for i in range(n):
        data = input().split()
        points[data[0]] = int(data[1])
        multiset.append(data[0])

    population = [input() for i in range(s)]

    trie = TrieDict('dict.txt')
    best_word, best_value = ag(t, trie, points, multiset, population, pop_size=len(points) * 20, parent_pop_size=len(points)*5)

    print(best_value)
    print(best_word,file=sys.stderr)