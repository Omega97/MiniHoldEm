"""
Simplified Texas Hold'em game.

SB = 1
BB = 2
Each player can call (1), raise (2), or fold (3).
After both players call, the player with the highest card wins.
"""
import numpy as np
import hashlib
from copy import deepcopy


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
        return [f'{c}{seeds[-1]}' for c in numbers[-n_cards:]]
    elif n_cards <= 13 * 2:
        return half_cards[-n_cards:]
    elif n_cards <= 13 * 4:
        return cards[-n_cards:]
    else:
        return [f'{i+1}🃏' for i in range(n_cards)]


class Game:
    ACTION_EMOJIS = np.array(['🤚', '📞', '🔼'])
    possible_actions = np.array(["fold", "call", "raise"])

    def __init__(self, n_players, n_cards, power=7, do_move_dealer=True, random_state=None):
        self.n_players = n_players
        self.n_cards = n_cards
        self.card_palette = card_palette(n_cards)
        self.power = power
        self.do_move_dealer = do_move_dealer
        self.n_chips = 2 ** power
        self.random_state = random_state
        self.verbose = False
        self.n_players = n_players
        self.n_players = self.n_players
        assert self.n_players > 1
        assert self.n_chips > 2
        assert self.n_cards >= self.n_players + (self.n_players == 2)

        self.hands_count = 0
        self.total_rewards = np.zeros(self.n_players)

        self.players = None
        self.deck = None
        self.hands = None
        self.stacks = None
        self.bets = None
        self.actions = None
        self.actors = None
        self.ply = None
        self.current_player = None
        self.folded = None
        self.dealer_id = None
        self.small_blind_id = None
        self.big_blind_id = None
        self.utg_id = None
        self.is_game_over = None
        self.winner = None

    def print(self, text):
        if self.verbose:
            print(text)

    def reset_game(self):
        """Reset game state."""
        np.random.seed(self.random_state)
        if self.random_state is not None:
            self.random_state += 1

        self.deck = np.arange(self.n_cards)
        np.random.shuffle(self.deck)
        self.stacks = np.ones(self.n_players, dtype=int) * self.n_chips
        self.bets = np.zeros(self.n_players, dtype=int)
        self.actions = []
        self.actors = []
        self.ply = 0
        self.folded = np.zeros(self.n_players, dtype=bool)
        if self.do_move_dealer:
            self.dealer_id = self.hands_count % self.n_players
        else:
            self.dealer_id = 0
        self.is_game_over = False

        self.blinds()
        self.deal()

    def print_card(self, card):
        """print card using the card palette."""
        return self.card_palette[card]

    def _bet(self, player, amount):
        """Player bets an amount."""
        self.stacks[player] -= amount
        self.bets[player] += amount
        if self.stacks[player] < 0:
            raise ValueError('Player does not have enough chips')

    def blinds(self):
        """Small blind and big blind."""
        if self.n_players == 2:
            self.small_blind_id = self.dealer_id
            self.big_blind_id = 1 - self.dealer_id
        else:
            self.small_blind_id = (self.dealer_id + 1) % self.n_players
            self.big_blind_id = (self.dealer_id + 2) % self.n_players
        self._bet(self.small_blind_id, 1)
        self._bet(self.big_blind_id, 2)
        self.utg_id = (self.big_blind_id + 1) % self.n_players
        self.current_player = self.utg_id
        self.print('\n>>> Blinds:')
        self.print(f'Button: Player {self.dealer_id} 🔘')
        self.print(f'    SB: Player {self.small_blind_id} 🟡')
        self.print(f'    BB: Player {self.big_blind_id} 🟡🟡')
        self.print(f'   UTG: Player {self.utg_id} 🔫')

    def deal(self):
        """Deal one card to each player."""
        self.hands = self.deck[:self.n_players]
        self.print('')
        for i in range(len(self.players)):
            if i == self.utg_id:
                _utg = f'(utg)'
            else:
                _utg = ''
            self.print(f'>>> Player {i} is dealt a {self.print_card(self.hands[i])}  {_utg}')
            self.players[i].start_game(self.get_personal_state(i))

    def max_bet(self):
        """Maximum bet so far."""
        return self.bets.max()

    def fold(self, player):
        """Player folds."""
        self.actions.append(0)
        self.actors.append(player)
        self.folded[player] = True
        card = self.print_card(self.hands[player])
        self.print(f'\n>>> Player {player} folds {Game.ACTION_EMOJIS[0]}  ({card})')

    def call(self, player):
        """Player calls the bet."""
        max_bet = self.max_bet()
        amount = max_bet - self.bets[player]
        self._bet(player, amount)
        self.actions.append(1)
        self.actors.append(player)
        card = self.print_card(self.hands[player])
        self.print(f'\n>>> Player {player} calls {max_bet} {Game.ACTION_EMOJIS[1]}  ({card})')

    def raise_(self, player):
        """Player raises the bet by double the current bet."""
        new_max_bet = 2 * self.max_bet()
        amount = new_max_bet - self.bets[player]
        self._bet(player, amount)
        self.actions.append(2)
        self.actors.append(player)
        card = self.print_card(self.hands[player])
        self.print(f'\n>>> Player {player} raises to {new_max_bet} {Game.ACTION_EMOJIS[2]}  ({card})')

    def get_reward(self):
        if self.winner is None:
            return np.zeros(self.n_players)
        v = self.stacks - self.n_chips
        assert v.sum() == 0, f'{v} must sum to 0, not {np.sum(v)} ({v})'
        return v

    def reset_fitness(self):
        self.total_rewards = 0
        self.hands_count = 0

    def get_fitness(self):
        return self.total_rewards / self.hands_count

    def get_legal_actions(self):
        """Legal actions for the current player (0=fold, 1=call, 2=raise)."""
        if self.is_game_over:
            raise ValueError('Game is over')

        v = np.ones(3, dtype=bool)
        current_bet = self.bets[self.current_player]

        # can't fold if the player already bet the max
        if current_bet == self.max_bet():
            v[0] = False

        # can't call if the player has no chips
        ...

        # can't raise if the player has no chips
        min_raise = 2 * self.max_bet() - current_bet
        if self.stacks[self.current_player] < min_raise:
            v[2] = False

        assert np.sum(v) > 0, f'No legal actions for player {self.current_player} {v}'  # todo check

        return v

    def _update_current_player(self):
        """Increase the id of current player to the next player still in game."""
        while True:
            self.current_player = (self.current_player + 1) % self.n_players
            if not self.folded[self.current_player]:
                break

    def _get_player_action(self) -> int:
        """Get action from player."""
        if self.is_game_over:
            raise ValueError('Game is over')
        legal_actions = self.get_legal_actions()
        # self.print(f'({", ".join(Game.ACTION_EMOJIS[legal_actions])})')
        player = self.players[self.current_player]
        state = self.get_personal_state(self.current_player)
        action = player(state, verbose=self.verbose)
        assert legal_actions[action], f'Invalid action {action}'
        return action

    def _execute_player_action(self, action):
        if action == 0:
            self.fold(self.current_player)
        elif action == 1:
            self.call(self.current_player)
        elif action == 2:
            self.raise_(self.current_player)
        else:
            raise ValueError(f'Invalid action {action}')

        self.print('')
        self.print(f'stacks = {self.stacks}')
        self.print(f'  bets = {self.bets}')
        self.print(f'   pot = {self.get_pot()} 🟡')
        self.ply += 1

    def _check__game_over(self):

        # check if only one player is still in the game
        if self.folded.sum() == self.n_players - 1:
            return True

        # check if all players have called
        bets = self.bets[~self.folded]
        if bets.max() == bets.min():
            # if all players saw action, the round is over
            if self.ply >= self.n_players:
                return True

        return False

    def game_loop(self):
        """"""
        while True:
            p = self.current_player
            self.print(f'\n>>> Action on Player {p} 💥')
            action = self._get_player_action()
            self._execute_player_action(action)
            self._update_current_player()

            # check if the round is over
            if self._check__game_over():
                self.is_game_over = True
                break

    def declare_winner(self):
        self.winner = np.argmax((self.hands + 1) * (1 - self.folded))
        chips = self.get_pot()
        self.stacks[self.winner] += chips

        if self.folded.sum() == self.n_players - 1:
            self.print(f"\n>>> Player {self.winner} didn't fold ✌️")
        else:
            self.print('\n>>> Showdown! 🃏')
            for i in range(self.n_players):
                if not self.folded[i]:
                    self.print(f' Player {i} shows {self.print_card(self.hands[i])}')
        card = self.print_card(self.hands[self.winner])
        self.print(f'\n>>> Player {self.winner} wins the hand! 🏆 ({card})')
        self.print(f'>>> Player {self.winner} wins {chips} 🟡')

    def get_game_record(self):
        """Game record"""
        return {'dealer': self.dealer_id,
                'actions': self.actions,
                'rewards': self.get_reward(),
                }

    def get_pot(self):
        return self.bets.sum()

    def get_current_player_position(self):
        """
        Position of the current player relative to the dealer.
        (not the player_id)
        """
        position = (self.current_player - self.utg_id) % self.n_players
        # assert position == self.ply % self.n_players
        return position

    def _common_info(self):
        info = {'decision_hash': self.decision_hash(),
                'position': self.get_current_player_position(),
                'n_players': int(self.n_players),
                'n_cards': int(self.n_cards),
                'n_chips': int(self.n_chips),
                'dealer': int(self.dealer_id),
                'sb': int(self.small_blind_id),
                'bb': int(self.big_blind_id),
                'utg': int(self.utg_id),
                'actions': tuple(self.actions),
                'actors': tuple(self.actors),
                'ply': int(self.ply),
                'stacks': tuple(self.stacks.tolist()),
                'bets': tuple(self.bets.tolist()),
                'folded': tuple(self.folded.tolist()),
                'pot': int(self.bets.sum()),
                'max_bet': int(self.max_bet()),
                'is_game_over': self.is_game_over,
                'reward': self.get_reward() if self.is_game_over else None,
                'legal_actions': self.get_legal_actions() if not self.is_game_over else None,
                }
        return deepcopy(info)

    def get_state(self):
        """Complete game state"""
        assert self.current_player is not None, 'Game not started'
        info = self._common_info()
        info['hands'] = self.hands
        info['winner'] = self.winner
        return deepcopy(info)

    def get_personal_state(self, player):
        """Abstract state representation from player's POV."""
        info = self._common_info()
        info['hole'] = self.hands[player]
        info['to_call'] = self.max_bet() - self.bets[player]
        info['player_id'] = player   # the id of the player (not the position)
        return deepcopy(info)

    def assign_reward(self):
        self.print(f'\n>>> Rewards 💰')
        reward = self.stacks - self.n_chips
        self.total_rewards += reward
        for i in range(len(self.players)):
            self.print(f' Player{i:2}) {reward[i]:+5.0f}')
        self.print(f'\n>>> Hand ends 🎊')
        self.print('*' * 40 + '\n')

        # pass rewards to agents
        for i, agent in enumerate(self.players):
            agent.end_game(reward[i], global_state=self.get_state())
        return reward

    def play(self, players, n_games=1, verbose=False):
        """Play the game."""
        assert len(players) == self.n_players, f'{len(players)} players != {self.n_players}'
        self.players = players
        self.verbose = verbose

        for _ in range(n_games * self.n_players):
            self.print('\n\n' + '*' * 40)
            self.print('>>> New hand starts 🎉')
            self.print(f'({self.n_players} players, {self.n_cards} cards, {self.n_chips} chips)')
            self.hands_count += 1
            self.reset_game()
            self.game_loop()
            self.declare_winner()
            self.assign_reward()

    def decision_hash(self, length=16) -> str:
        """Hash of the (move tree) state with sha256"""
        t = tuple(self.actions)
        enc = str(t).encode()
        h = hashlib.sha256(enc).hexdigest()
        return h[:length]
