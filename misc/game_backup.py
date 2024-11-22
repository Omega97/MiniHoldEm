"""
Simplified Texas Hold'em game.

SB = 1
BB = 2
Each player can call (1), raise (2), or fold (3).
After both players call, the player with the highest card wins.
"""
import numpy as np


def one_hot(n, length, dtype=None):
    out = np.zeros(length, dtype=dtype)
    out[n] = 1
    return out


def card_palette(n_cards):
    seeds = ['♥️', '♦️', '♣️', '♠️']
    numbers = [f'{i}' for i in range(2, 10)] + ['T', 'J', 'Q', 'K', 'A']
    half_cards = [f'{n}{s}' for s in numbers for n in [seeds[3], seeds[0]]]
    cards = [f'{n}{s}' for s in numbers for n in reversed(seeds)]
    if n_cards <= 13:
        return [f'{c}🃏' for c in numbers[-n_cards:]]
    elif n_cards <= 13 * 2:
        return half_cards[-n_cards:]
    elif n_cards <= 13 * 4:
        return cards[-n_cards:]
    else:
        return [f'{i+1}🃏' for i in range(n_cards)]


class Game:

    def __init__(self, players, n_cards, n_rounds, random_state=None):
        self.players = players
        self.n_cards = n_cards
        self.card_palette = card_palette(n_cards)
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.n_players = len(players)
        self.n_plys = n_rounds * self.n_players
        self.n_players = self.n_players
        self.initial_chips = 2 ** (self.n_plys + 1)
        assert self.n_players > 1
        assert self.n_cards >= self.n_players + (self.n_players==2)
        assert self.n_rounds > 1

        self.deck = None
        self.hands = None
        self.chips = None
        self.bets = None
        self.actions = None  # none, call, raise, fold
        self.ply = None
        self.dealer = None
        self.current_player = None
        self.folded = None

    def reset(self):
        """Reset game state."""
        np.random.seed(self.random_state)
        self.random_state += 1
        self.deck = np.arange(self.n_cards)
        np.random.shuffle(self.deck)
        self.chips = np.ones(self.n_players, dtype=int) * self.initial_chips
        self.bets = np.zeros(self.n_players, dtype=int)
        self.actions = np.zeros(self.n_plys, dtype=int)
        self.ply = 0
        self.dealer = np.random.randint(self.n_players)
        self.folded = np.zeros(self.n_players, dtype=bool)

    def print_card(self, card):
        """print card using the card palette."""
        return self.card_palette[card]

    def _bet(self, player, amount):
        """Player bets an amount."""
        self.chips[player] -= amount
        self.bets[player] += amount
        if self.chips[player] < 0:
            raise ValueError('Player does not have enough chips')

    def blinds(self):
        """Small blind and big blind."""
        i_small = self.n_players - 2 if self.n_players > 2 else 1
        i_small = (i_small + self.dealer) % self.n_players
        i_big = self.n_players - 1 if self.n_players > 2 else 0
        i_big = (i_big + self.dealer) % self.n_players
        self._bet(i_small, 1)
        self._bet(i_big, 2)
        utg = (i_big + 1) % self.n_players
        self.current_player = utg
        print('\n>>> Blinds:')
        print(f'Button: Player {self.dealer} 🔘')
        print(f'    SB: Player {i_small} 🪙')
        print(f'    BB: Player {i_big} 🪙🪙')
        print(f'   UTG: Player {utg} 🔫')

    def deal(self):
        """Deal one card to each player"""
        self.hands = self.deck[:self.n_players]
        print()
        for i in range(len(self.players)):
            print(f'>>> Player {i} is dealt a {self.print_card(self.hands[i])}')

    def call(self, player):
        """Player calls the bet."""
        print(f'Player {player} calls 🤙')
        amount = self.bets.max() - self.bets[player]
        self._bet(player, amount)
        self.actions[self.ply] = 1

    def raise_(self, player):
        """Player raises the bet by double the current bet."""
        print(f'Player {player} raises 🤚')
        amount = 2 * self.bets.max() - self.bets[player]
        self._bet(player, amount)
        self.actions[self.ply] = 2

    def fold(self, player):
        """Player folds."""
        print(f'>>> Player {player} folds 👎')
        self.actions[self.ply] = 3
        self.folded[player] = True

    def reward(self):
        v = self.chips / self.initial_chips - 1
        assert np.sum(v) == 0, f'{v} must sum to 0.'
        return v

    def get_legal_actions(self):
        """Legal actions for the current player (0=fold, 1=call, 2=raise)."""
        return np.ones(3, dtype=bool)

    def game_loop(self):
        actions = np.array(["fold", "call", "raise"])

        while True:
            print(f'\n>>> Action on Player {self.current_player} 💥')
            legal_actions = self.get_legal_actions()
            print(f'Legal actions: {", ".join(actions[legal_actions].tolist())}')

            while True:
                self.current_player = (self.current_player + 1) % self.n_players
                if not self.folded[self.current_player]:
                    break

            action = self.players[self.current_player](self.get_personal_state(self.current_player))
            assert legal_actions[action]

            (self.fold, self.call, self.raise_)[action](self.current_player)

            # check if only one player is still in the game
            if self.folded.sum() == self.n_players - 1:
                break

            self.ply += 1
            if self.ply == self.n_plys:
                break
        winner = np.argmax((self.hands + 1) * (1 - self.folded))
        print(f'\n>>> Player {winner} wins the hand! 🏆')
        chips = self.bets.sum()
        print(f'>>> Player {winner} wins {chips} 🪙')
        self.chips[winner] += chips

    def get_state(self):
        """Complete game state"""
        return {'hands': self.hands,
                'chips': self.chips,
                'bets': self.bets,
                'legal_actions': self.get_legal_actions(),
                'actions': self.actions,
                'ply': self.ply,
                }


    def get_personal_state(self, player):
        """Abstract state representation from player's POV."""
        return {'hand': self.hands[player],
                'chips': self.chips,
                'bets': self.bets,
                'legal_actions': self.get_legal_actions(),
                'actions': self.actions,
                'ply': self.ply,
                }

    def play(self):
        """Play a game."""
        print('\n>>> New hand starts 🎉')
        print(f'({self.n_players} players, {self.n_cards} cards, {self.n_rounds} rounds)')
        self.reset()
        self.blinds()
        self.deal()
        print(f'\n>>> Game state 🎲')
        print('\n'.join([f"{k:>13}: {v}" for k, v in self.get_state().items()]))
        self.game_loop()
        print(f'\n>>> Rewards 💰')
        rewards = self.reward()
        for i in range(len(self.players)):
            print(f' Player{i:2}) {rewards[i]:+7.2%}')
        print(f'\n>>> Hand ends 🎊')
        print()
        return self.reward()


def test():

    def random_agent():
        def wrap(state):
            """0=fold, 1=call, 2=raise"""
            legal_actions = state['legal_actions']
            p = np.ones(len(legal_actions)) / len(legal_actions)
            p *= legal_actions
            return np.random.choice(len(legal_actions), p=p)
        return wrap

    players = [random_agent(), random_agent(), random_agent(), random_agent()]
    game = Game(players, n_cards=6, n_rounds=2, random_state=0)
    game.play()


if __name__ == '__main__':
    test()
