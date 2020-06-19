import numpy as np
import random
import copy

''' 
BLOCK = (x,y,a,b,intensity,(i,j))
x,y - współrzedne
a,b - wymiary
intensity - wartosc
(i,j) - klucz
'''


class Image:
    def __init__(self, M, n, m, k):
        self.M = M
        self.original_image = copy.deepcopy(self.M)
        # vertical dim
        self.n = n
        # horizontal dim
        self.m = m
        # blocksize
        self.k = k

        self.fit_perfectly = self.n % self.k == 0 and self.m % self.k == 0

        self.VALUES = [0, 32, 64, 128, 160, 192, 223, 255]

        self.blocks = {}
        self.initialize_blocks()

    def get_distance_between(self, M1, M2):
        return ((M1 - M2) ** 2).mean()

    def initialize_blocks(self):
        x = 0
        y = 0

        # ile rzedów bloków zmieści się w macierzy
        blockrows_in_matrix = self.n // self.k

        # ile kolumn bloków zmieści się w macierzy
        blockcols_in_matrix = self.m // self.k

        # blocks = []
        blocks = {}
        M = copy.deepcopy(self.original_image)

        for i in range(blockrows_in_matrix):
            for j in range(blockcols_in_matrix):
                if j == blockcols_in_matrix - 1 and i == blockrows_in_matrix - 1:
                    block = M[x:, y:]
                elif j == blockcols_in_matrix - 1:
                    block = M[x:x + self.k, y:]
                elif i == blockrows_in_matrix - 1:
                    block = M[x:, y:y + self.k]
                else:
                    block = M[x:x + self.k, y:y + self.k]
                blocks[(i, j)] = (x, y, block.shape[0], block.shape[1], self.VALUES[3], (i, j))
                y += self.k

            y = 0
            x += self.k

        self.blocks = blocks

    def find_matching_blocks_neighbours(self, block, blocks):
        ''' return neighbours of block from blocks
            neighbours shares one full side
            '''
        x, y, a, b, intensity, (i, j) = block

        candidates = {'U': blocks.get((i - 1, j), None),
                      'R': blocks.get((i, j + 1), None),
                      'L': blocks.get((i, j - 1), None),
                      'D': blocks.get((i + 1, j), None)}

        neighbours = []

        for candidate_key, candidate in candidates.items():
            if candidate:
                xx, yy, aa, bb, intensity, (ii, jj) = candidate
                if candidate_key in ['D', 'R'] and yy == y - (j - jj) * b and xx == x - (i - ii) * a:
                    neighbours.append(candidate)
                if candidate_key in ['U', 'L'] and yy == y - (j - jj) * bb and xx == x - (i - ii) * aa:
                    neighbours.append(candidate)

        return neighbours

    def get_block_distance(self, block):
        ''' return distance from original matrix on specific block '''
        x, y, a, b, intensity, (i, j) = block
        original_block = self.original_image[x:x + a, y:y + b]
        block = intensity * np.ones((a, b), dtype=int)
        return self.get_distance_between(original_block, block)

    def get_matrix_from_blocks(self, blocks):
        ''' return whole matrix builded with blocks '''
        M = np.ones((self.n, self.m), dtype=int)

        for block_key, block in blocks.items():
            x, y, a, b, intensity, (i, j) = block
            new_block = intensity * np.ones((a, b), dtype=int)
            M[x:x + a, y:y + b] = new_block
        return M

    def get_neighbour_solution(self, blocks):
        ''' returns neighbour solution'''

        # jesli obraz jest podzielny przez reszty na bloki to zamiany nie maja sensu
        if not self.fit_perfectly:
            chance = random.random()
            if chance < 0.01:
                blocks = self.swap_between_random_matching_neighbours(blocks)
            elif chance < 0.2:
                blocks = self.spread_random_neighbours(blocks)

        random_block_key, random_block = random.choice(list(blocks.items()))
        blocks[random_block_key] = self.set_best_intensity_in_block(random_block)

        return blocks

    def swap_blocks(self, block1, block2, blocks):
        ''' swap between blocks '''
        x1, y1, a1, b1, intensity1, (i1, j1) = block1
        x2, y2, a2, b2, intensity2, (i2, j2) = block2

        if x1 == x2 and y1 == y2:
            return blocks
        if x1 != x2 and y1 != y2:
            return blocks

        # block2 jest z dołu
        if i1 < i2:
            block2 = x1, y1, a2, b2, intensity2, (i1, j1)
            block1 = x1 + a2, y2, a1, b1, intensity1, (i2, j2)
        # block2 jest z gory
        if i1 > i2:
            block2 = x2 + a1, y1, a2, b2, intensity2, (i1, j1)
            block1 = x2, y2, a1, b1, intensity1, (i2, j2)

        # block 2 jest z prawej
        if j1 < j2:
            block2 = x1, y1, a2, b2, intensity2, (i1, j1)
            block1 = x1, y1 + b2, a1, b1, intensity1, (i2, j2)

        # block 2 jest z lewej
        if j1 > j2:
            block2 = x1, y2 + b1, a2, b2, intensity2, (i1, j1)
            block1 = x2, y2, a1, b1, intensity1, (i2, j2)

        blocks[block2[5]] = block2
        blocks[block1[5]] = block1
        return blocks

    def swap_between_random_matching_neighbours(self, blocks):
        ''' swaps two matching blocks '''

        if self.fit_perfectly:
            return blocks

        random_block_key, random_block = random.choice(list(blocks.items()))
        matching_neighbours = self.find_matching_blocks_neighbours(random_block, blocks)

        while random_block[2] == self.k and random_block[3] == self.k:
            random_block_key, random_block = random.choice(list(blocks.items()))
            matching_neighbours = self.find_matching_blocks_neighbours(random_block, blocks)

        if len(matching_neighbours) > 0:
            random.shuffle(matching_neighbours)
            blocks = self.swap_blocks(random_block, matching_neighbours[0], blocks)
            return blocks

        return blocks

    def set_random_intensity_in_block(self, block):
        x, y, a, b, intensity, (i, j) = block
        return x, y, a, b, random.choice(self.VALUES), (i, j)

    def set_best_intensity_in_block(self, block):
        x, y, a, b, intensity, (i, j) = block
        candidates = [self.get_block_distance((x, y, a, b, self.VALUES[s], (i, j))) for s in range(len(self.VALUES))]
        mini = min(candidates)
        return x, y, a, b, self.VALUES[candidates.index(mini)], (i, j)

    def spread_random_neighbours(self, blocks):
        if self.fit_perfectly:
            return blocks

        random_block_key, random_block = random.choice(list(blocks.items()))
        matching_neighbours = self.find_matching_blocks_neighbours(random_block, blocks)

        while random_block[2] == self.k and random_block[3] == self.k:
            random_block_key, random_block = random.choice(list(blocks.items()))
            matching_neighbours = self.find_matching_blocks_neighbours(random_block, blocks)

        if len(matching_neighbours) > 0:
            random.shuffle(matching_neighbours)
            blocks = self.spread_between_blocks(random_block, matching_neighbours[0], blocks)
            return blocks

        return blocks

    def spread_between_blocks(self, block1, block2, blocks):
        '''
            block1 giver
            block2 receiver
        '''
        x1, y1, a1, b1, intensity1, (i1, j1) = block1
        x2, y2, a2, b2, intensity2, (i2, j2) = block2

        if x1 == x2 and y1 == y2:
            return blocks
        if x1 != x2 and y1 != y2:
            return blocks
        if max(a1, b1, a2, b2) == self.k:
            return blocks

        # jeśli ma co oddać
        if a1 > self.k:
            to_give = random.randint(1, a1 - self.k)

            # block2 jest z dołu
            if i1 < i2:
                block2 = x2 - to_give, y2, a2 + to_give, b2, intensity2, (i2, j2)
                block1 = x1, y1, a1 - to_give, b1, intensity1, (i1, j1)
            # block2 jest z gory
            if i1 > i2:
                block2 = x2, y2, a2 + to_give, b2, intensity2, (i2, j2)
                block1 = x1 + to_give, y1, a1 - to_give, b1, intensity1, (i1, j1)

        if b1 > self.k:
            to_give = random.randint(1, b1 - self.k)
            # block 2 jest z prawej
            if j1 < j2:
                block2 = x2, y2 - to_give, a2, b2 + to_give, intensity2, (i2, j2)
                block1 = x1, y1, a1, b1 - to_give, intensity1, (i1, j1)

            # block 2 jest z lewej
            if j1 > j2:
                block2 = x2, y2, a2, b2 + to_give, intensity2, (i2, j2)
                block1 = x1, y1 + to_give, a1, b1 - to_give, intensity1, (i1, j1)

        blocks[block2[5]] = block2
        blocks[block1[5]] = block1
        return blocks

    def blocks_legal_test(self, blocks):
        ''' test to check if given image is consisten with given rules'''
        for block_key, block in blocks.items():
            if block[2] < self.k or block[3] < self.k:
                return False
            if not block[4] in self.VALUES:
                return False
        return True
