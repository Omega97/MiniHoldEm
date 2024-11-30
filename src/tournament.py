import os.path
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from fitness import Fitness
from src.game import Game
from src.agents import TrainingAgent, load_game_tree, AGENTS_DIR


def softmax(logits):
    mat = np.exp(logits)
    v = np.sum(mat, axis=1)
    v = np.expand_dims(v, axis=0).T
    return mat / v


def compute_entropy(logits):
    """ Computes the entropy of a given matrix of logits."""
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
    return np.average(entropy)


class Tournament:
    RECORD = []
    ENTROPY = []
    AGENT_COUNT = 0
    EPOCHS = 0
    FITNESS_DATABASE = dict()

    def __init__(self, game: Game, agents: list, fitness, p_mutation=0.1):

        # environment
        self.game = game
        self.n_players = game.n_players
        self.n_cards = game.n_cards
        self.power = game.power
        self.p_mutation = p_mutation
        self.game_tree = load_game_tree(self.n_players, self.power)

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

    def get_best_agent(self) -> TrainingAgent:
        return self.agents[-1]

    def _compute_fitness(self):
        """Compute fitness
        Use the best agent to evaluate all the other agents
        """
        self.scores = np.zeros(self.n_agents)
        self.counts = np.zeros(self.n_agents, dtype=int)

        best_agent = self.get_best_agent()
        for i in range(self.n_agents-1):
            agents = tuple([self.agents[i]] + [best_agent] * (self.n_players-1))
            f_ = self.fitness(agents)
            self.add_score(i, f_[0])
        self.add_score(self.n_agents-1, 0.)

    def _sort_agents(self):
        """Sort players by score"""
        order = np.argsort(self.get_fitness())
        self.agents = [self.agents[i] for i in order]
        self.scores = self.scores[order]
        self.counts = self.counts[order]
        self.agent_ids = self.agent_ids[order]

    def mutation(self, i_mutate, sigma):
        shape = self.agents[i_mutate].logits.shape
        dx = np.random.normal(0, sigma, shape)
        while True:
            mask = np.random.random(dx.shape) < self.p_mutation
            if np.sum(mask):
                break
        self.agents[i_mutate].logits += dx * mask

    def _replacement(self, i_keep: list, i_replace: list, sigma=1.):
        assert len(i_keep) == len(i_replace)
        for i, j in zip(i_keep, i_replace):
            print(f'{i} > {j}   {self.agent_ids[i]:3} -> {self.agent_ids[j]:3}')
            # replacement
            self.reset_agent(j, deepcopy(self.agents[i]))
            # mutation
            self.mutation(j, sigma)

    def _crossover(self, i_keep: list, i_replace: list, w=.5):
        assert len(i_keep) == len(i_replace)
        for i, j in zip(i_keep, i_replace):
            print(f'{i} > {j}   {self.agent_ids[i]:3} + {self.agent_ids[j]:3} -> {self.agent_ids[j]:3}')
            self.reset_agent(j, deepcopy(self.agents[i]))
            new_logits = w * self.agents[i].logits + (1-w) * self.agents[j].logits
            self.agents[j].logits = new_logits

    def evaluate(self, benchmark_agent, top_n=5):
        """evaluation"""
        print()
        w = np.exp(np.arange(top_n) / (top_n-1) * 4)
        w /= np.sum(w)

        v = []
        for i in reversed(range(self.n_agents - top_n, self.n_agents)):
            agents = [self.agents[i]] + [benchmark_agent] * (self.n_players-1)
            f_ = self.fitness(agents)
            v.append(f_[0])
            print(f'{self.agent_ids[i]:4}) {f_[0]:8.3f}')
        v = np.array(v)
        score = v.dot(w)
        Tournament.RECORD.append(score)

        v = []
        for i in reversed(range(self.n_agents - top_n, self.n_agents)):
            logits = self.agents[i].logits
            s = compute_entropy(logits) / np.log(3)
            v.append(s)
        v = np.array(v)
        s = v.dot(w)
        Tournament.ENTROPY.append(s)

    def run(self, n_epochs, n_replace, n_crossover,
            benchmark_agent, sigma=0.1, top_n=5, agent_path=None):
        for epoch in range(n_epochs):
            Tournament.EPOCHS += 1
            print(f'\nEpoch {Tournament.EPOCHS}   ({self.n_agents}/{self.AGENT_COUNT} angents)')
            if Tournament.EPOCHS > 1:

                self._crossover(i_keep=list(range(self.n_agents-n_crossover, self.n_agents)),
                                i_replace=list(range(n_crossover)),
                                w=0.5 + np.random.random() ** 2 * 0.8)
                self._replacement(i_keep=list(range(self.n_agents-n_replace, self.n_agents)),
                                  i_replace=list(range(1, 1+n_replace)),
                                  sigma=sigma)

            self._compute_fitness()
            self._sort_agents()
            self.evaluate(benchmark_agent, top_n)

            print()
            logits = self.get_best_agent().logits
            norm = np.mean(logits**2)**0.5
            print(f'   norm = {norm:.3f}')
            s = compute_entropy(logits) / np.log(3)
            print(f'entropy = {s:.3f}')

            if agent_path:
                self.get_best_agent().save(agent_path)


def main(n_players=3, n_cards=6, power=4,
         n_agents=10, n_epochs=30, n_replace=3, n_crossover=1,
         init_scale=.1, sigma=.3, top_n=3,
         reg=1e-3, p_mutation=0.1,
         seed=None, random_state=None,
         ):

    np.random.seed(seed)
    path = AGENTS_DIR + f'best_agent_{n_players}_{n_cards}_{power}.pkl'
    game = Game(n_players=n_players, n_cards=n_cards, power=power, random_state=random_state)

    # create agents
    if os.path.exists(path):
        # load pre-trained agents
        agents = [TrainingAgent(game).load(path)
                  for _ in range(n_agents)]
    else:
        agents = [TrainingAgent(game) for _ in range(n_agents)]

    # fitness
    fitness = Fitness(n_players, n_cards, TrainingAgent.GAME_TREE, reg=reg)

    # tournament
    tournament = Tournament(game, agents, fitness, p_mutation=p_mutation)
    for i in range(n_agents):
        tournament.mutation(i, init_scale)

    # benchmark agent
    try:
        benchmark_agent = TrainingAgent(game).load(path)
    except FileNotFoundError:
        benchmark_agent = TrainingAgent(game, sigma=0.)

    tournament.run(n_epochs=n_epochs, n_replace=n_replace, n_crossover=n_crossover,
                   benchmark_agent=benchmark_agent, sigma=sigma, top_n=top_n,
                   agent_path=path)

    fig, ax = plt.subplots(ncols=2)

    plt.sca(ax[0])
    plt.plot(tournament.RECORD)
    plt.title(f'Mini Hold\'em ({n_players} players, {n_cards} cards, {2**power} chips)')
    plt.xlabel('epoch')
    plt.ylabel('fitness')

    plt.sca(ax[1])
    plt.plot(tournament.ENTROPY, c='green')
    plt.title('Entropy')
    plt.xlabel('epoch')
    plt.ylabel('entropy')

    plt.show()


if __name__ == '__main__':
    main()
