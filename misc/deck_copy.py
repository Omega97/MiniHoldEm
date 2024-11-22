import numpy as np
import matplotlib.pyplot as plt
from time import time


class Simulation:

    def __init__(self, deck, hand_sizes, n_epochs=1_000):
        self.deck = deck
        self.hand_sizes = hand_sizes
        self.n_epochs = n_epochs
        self.elements = tuple(set([int(i) for i in self.deck]))
        self.deck_mat = None
        self.cumul = dict()
        self.counts = dict()
        self.presence = dict()

        self._compute_deck_mat()
        self._compute_cumul()
        self._compute_all_counts()
        self._compute_presence()

    def _compute_deck_mat(self):
        """
        Create a matrix that represents 'n_epochs'
        shuffled copies of the deck.
        """
        self.deck_mat = np.tile(self.deck, (self.n_epochs, 1))
        self.deck_mat = [np.random.permutation(self.deck_mat[0]) for _ in range(self.n_epochs)]
        self.deck_mat = np.array(self.deck_mat)

    def _compute_cumul(self):
        """
        Compute the cumulative of 'deck_mat'
        """
        self.cumul = dict()
        for e in self.elements:
            self.cumul[e] = np.cumsum(self.deck_mat == e, axis=1)

    def get_cumul(self, element):
        if element in self.cumul:
            return self.cumul[element].copy()
        else:
            # return a matrix of zeros
            return np.zeros((self.n_epochs, len(self.deck)), dtype=int)

    def _compute_count(self, element, hand_size):
        """Compute the number of cards in hand"""
        hand = self.get_cumul(element)
        t = (hand[:, hand_size - 1:hand_size],
             hand[:, hand_size:] - hand[:, :-hand_size])
        hand = np.concatenate(t, axis=1)
        return hand

    def _compute_all_counts(self):
        """
        Compute the number of cards in hand
        for each element and hand size
        """
        self.counts = dict()
        for e in self.elements:
            self.counts[e] = dict()
            for hand_size in self.hand_sizes:
                self.counts[e][hand_size] = self._compute_count(e, hand_size)

    def _compute_presence(self):
        """
        Compute the presence of each element in hand for each hand size
        (1 if the element is in that hand, 0 otw.)
        """
        self.presence = dict()
        for e in self.elements:
            self.presence[e] = dict()
            for hand_size in self.hand_sizes:
                self.presence[e][hand_size] = self.counts[e][hand_size] > 0

    def get_count(self, element, hand_size):
        if element in self.counts:
            if hand_size in self.counts[element]:
                return self.counts[element][hand_size].copy()
        # return a matrix of zeros
        return np.zeros(self.n_epochs, dtype=int)

    def get_presence(self, element, hand_size):
        if element in self.presence:
            if hand_size in self.presence[element]:
                return self.presence[element][hand_size].copy()
        # return a matrix of zeros
        return np.zeros((self.n_epochs, len(self.deck)), dtype=int)

    def run(self, hand_size):
        """compute the probability of having both cards in hand
        More efficient implementation, since we repeat the
        sampling process 'deck_size - hand_size + 1' times
        """
        assert hand_size in self.hand_sizes
        hand_1 = self.get_presence(1, hand_size)
        hand_2 = self.get_presence(2, hand_size)
        # hand_1 = self.get_count(1, hand_size) > 0
        # hand_2 = self.get_count(2, hand_size) > 0
        mat = hand_1 * hand_2
        return np.mean(mat)


def main(deck_size=40, hand_sizes=(4, 5, 6)):

    t = time()

    x_ = np.arange(0, 21)
    y = dict()

    # iterate through various parameters of the deck
    for i in range(len(x_)):

        # build deck
        deck = [1] * x_[i]
        deck += [2] * x_[i]
        deck += [0] * (deck_size - len(deck))
        deck = np.array(deck)
        assert len(deck) == deck_size

        # run simulation
        simulation = Simulation(deck, hand_sizes)
        for hand_size in hand_sizes:
            if hand_size not in y:
                y[hand_size] = []
            y[hand_size].append(simulation.run(hand_size))

    print(f'Time taken: {time()-t:.2f} s')

    # plot results
    for hand_size in hand_sizes:
        plt.plot(x_, y[hand_size], label=f'Hand size: {hand_size}')

    plt.title('Probability of having a 1 and a 2 in hand')
    plt.xlabel('Number of copies in the deck')
    plt.ylabel('Probability')
    plt.xticks(x_)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
