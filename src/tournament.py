import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from fitness import Fitness
from src.game import Game
from src.agents import TrainingAgent, load_game_tree


def softmax(logits):
    mat = np.exp(logits)
    v = np.sum(mat, axis=1)
    v = np.expand_dims(v, axis=0).T
    return mat / v


def entropy(logits):
    mat = np.exp(logits)
    return -np.sum(logits * mat, axis=1)


class Tournament:
    RECORD = []
    AGENT_COUNT = 0
    EPOCHS = 0
    FITNESS_DATABASE = dict()

    def __init__(self, game: Game, agents: list, fitness):

        # environment
        self.game = game
        self.game_tree = load_game_tree(game.n_players, game.power)
        self.n_cards = game.n_cards

        # initialization
        self.agents = agents
        self.n_agents = len(agents)
        Tournament.AGENT_COUNT += self.n_agents

        self.fitness = fitness
        self.scores = None
        self.agent_ids = np.arange(self.n_agents)

    def reset_agent(self, i, new_agent):
        self.agents[i] = new_agent
        self.scores[i] = 0.
        self.agent_ids[i] = Tournament.AGENT_COUNT
        Tournament.AGENT_COUNT += 1

    def add_score(self, i, value):
        self.scores[i] += value
        self.counts[i] += 1

    def get_fitness(self):
        return self.scores / self.counts

    @staticmethod
    def _save_fitness_to_db(i, j, f):
        key = (i, j)
        if key not in Tournament.FITNESS_DATABASE:
            Tournament.FITNESS_DATABASE[key] = f

    @staticmethod
    def _get_fitness_from_db(i, j):
        key = (i, j)
        if key in Tournament.FITNESS_DATABASE:
            return Tournament.FITNESS_DATABASE[key]

    def _compute_fitness(self):
        """Compute fitness by making pairwise matches"""
        self.scores = np.zeros(self.n_agents)
        self.counts = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            for j in range(i):
                agents = (self.agents[i], self.agents[j])
                f_ = self._get_fitness_from_db(self.agent_ids[i], self.agent_ids[j])
                if f_ is None:
                    f_ = self.fitness(agents)
                    self._save_fitness_to_db(self.agent_ids[i], self.agent_ids[j], f_)
                self.add_score(i, f_[0])
                self.add_score(j, f_[1])

    def _sort_agents(self):
        # sort players by score
        order = np.argsort(self.get_fitness())
        self.agents = [self.agents[i] for i in order]
        self.scores = self.scores[order]
        self.counts = self.counts[order]
        self.agent_ids = self.agent_ids[order]

    def _replacement(self, n_replace, sigma):

        indices = np.arange(self.n_agents)
        indices_replace = indices[:n_replace]
        indices_keep = indices[n_replace:]
        p = indices_keep / indices_keep.sum()
        indices_keep = np.random.choice(indices_keep, n_replace, replace=False, p=p)

        for i, j in zip(indices_keep, indices_replace):
            print(f'({i} -> {j})  {self.agent_ids[i]:3} -> {self.agent_ids[j]:3}')
            # replacement
            self.reset_agent(j, deepcopy(self.agents[i]))
            # mutation
            shape = self.agents[j].logits.shape
            dx = np.random.normal(0, sigma, shape)
            self.agents[j].logits += dx

    def _crossover(self, top_n=5):
        top_indices = np.arange(self.n_agents - top_n, self.n_agents)
        i_better, i_worst = np.random.choice(top_indices, 2, replace=False)

        i_replace = 0
        print(f'({i_better} + {i_worst} -> {i_replace})  '
              f'{self.agent_ids[i_better]:3} + {self.agent_ids[i_worst]:3} -> {self.agent_ids[i_replace]:3}')
        self.reset_agent(i_replace, deepcopy(self.agents[i_better]))
        self.agents[i_replace].logits = (self.agents[i_better].logits + self.agents[i_worst].logits)/2

    def get_best_agent(self) -> TrainingAgent:
        return self.agents[-1]

    def evaluate(self, top_n=5):

        # evaluation
        print()
        random_agent = TrainingAgent(self.game, sigma=0.)
        w = np.exp(np.arange(top_n) / (top_n-1) * 4)
        w /= np.sum(w)
        v = []
        for i in reversed(range(self.n_agents - top_n, self.n_agents)):
            agents = [self.agents[i], random_agent]
            f_ = self.fitness(agents)
            v.append(f_[0])
            print(f'{self.agent_ids[i]:4}) {f_[0]:8.3f}')
        v = np.array(v)
        score = v.dot(w)
        Tournament.RECORD.append(score)

    def run(self, n_epochs, n_replace, sigma=0.1, top_n=5):
        for epoch in range(n_epochs):
            Tournament.EPOCHS += 1
            print(f'\nEpoch {Tournament.EPOCHS}   ({self.n_agents}/{self.AGENT_COUNT} angents)')
            if Tournament.EPOCHS > 1:
                self._replacement(n_replace, sigma)
                self._crossover()
            self._compute_fitness()
            self._sort_agents()
            self.evaluate(top_n)


def main(n_agents=20, n_epochs=500, n_replace=2, init_scale=.5,
         sigma=.5, seed=0, random_state=0, top_n=10,
         n_players=2, n_cards=6, power=2):

    np.random.seed(seed)
    game = Game(n_players=n_players, n_cards=n_cards, power=power, random_state=random_state)
    agents = [TrainingAgent(game, sigma=init_scale) for _ in range(n_agents)]
    fitness = Fitness(n_players, n_cards, TrainingAgent.GAME_TREE)
    tournament = Tournament(game, agents, fitness)
    tournament.run(n_epochs=n_epochs, n_replace=n_replace, sigma=sigma, top_n=top_n)

    # save parameters
    tournament.get_best_agent().save(f'best_agent_{n_players}_{n_cards}_{power}.pkl')

    plt.plot(tournament.RECORD)
    plt.title('Mini Hold\'em EA')
    plt.xlabel('epoch')
    plt.ylabel('fitness')
    plt.show()


if __name__ == '__main__':
    main()
