import copy
import random

_end = '_end_'


class Trie:
    def __init__(self, filename):
        self.filename = filename

    def contains(self, word):
        pass

    def is_proper_word(self, word, multiset):
        if not self.contains(word):
            return False


class TrieList(Trie):
    def __init__(self, filename):
        super().__init__(filename)
        self.words = []
        with open(filename, 'r') as file:
            self.words = file.read().splitlines()

    def contains(self, word):
        return word in self.words


class TrieDict(Trie):
    def __init__(self, filename):
        super().__init__(filename)
        self.words = {}
        with open(filename, 'r') as file:
            lines = file.readlines()
            for word in lines:
                word = word.strip()
                current_dict = self.words
                for letter in word:
                    current_dict = current_dict.setdefault(letter, {})
                current_dict[_end] = _end

    def contains(self, word):
        ''' sprawdza czy słowo znajduje sie w słowniku'''
        current_dict = self.words
        for letter in word:
            if letter not in current_dict:
                return False
            current_dict = current_dict[letter]
        return _end in current_dict

    def get_possible_next_letters(self, prefix, current_dict=None):
        '''
        :param prefix: początek słowa dla którego szukamy nastepnych możliwych przejść
        :param current_dict: podzbiór zbioru wszystkich wyrazów
        :return: zwraca nastepne mozliwe litery, ktora droga mozemy isc
        '''
        if not current_dict:
            current_dict = self.words
        if prefix:
            if prefix[0] not in current_dict:
                return []
            for letter in prefix:
                if letter not in current_dict:
                    return []
                current_dict = current_dict[letter]
        next_letters = list(current_dict)
        if _end in next_letters:
            next_letters.remove(_end)
        return next_letters

    def get_random_from_prefix(self, prefix, multiset):
        ''' zwraca prefix(może być prawidłowym słowem) jakiegoś wyrazu ze słowniak zawierający podany prefix '''
        mset = copy.copy(multiset)
        word = prefix

        if prefix:
            # remove used letters
            for letter in prefix:
                mset.remove(letter)

        for i in range(len(mset)):
            next_letters = self.get_possible_next_letters(word)
            intersection = list(set(next_letters) & set(mset))
            if not intersection:
                return word
            else:
                random.shuffle(intersection)
                if self.contains(word) and random.random() < 0.01 * (len(word) ** 2):
                    return word
                letter = intersection.pop(0)
                mset.remove(letter)
                word += letter

        return word

    def get_random_possible_from_multiset(self, multiset):
        ''' multiset - zbiór dostępnych liter
            zwraca prefix(może być prawidłowym słowem) prawidłowego słowa, korzystając tylko z liter z multisetu
        '''
        random.shuffle(multiset)
        multiset = copy.copy(multiset)
        word = multiset.pop(0)
        for i in range(len(multiset)):
            next_letters = self.get_possible_next_letters(word)
            intersection = list(set(next_letters) & set(multiset))
            if not intersection:
                return word
            else:
                random.shuffle(intersection)
                if self.contains(word) and random.random() < 0.01 * (len(word) ** 2):
                    return word
                letter = intersection.pop(0)
                multiset.remove(letter)
                word += letter


if __name__ == '__main__':
    t = TrieDict('dict.txt')
    for i in range(100):
        word = t.get_random_from_prefix('',list('auneapbstdh'))
        print(word)