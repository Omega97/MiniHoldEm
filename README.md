
# Mini Hold'Em


This Program uses **evolutionary algorithms** and **gradient-based methods** to master a simplified version of Heads-up Limit poker Texas Hold'Em.

---

## Rules of the game

1. The Button bets the **Small Blind** (SB) of 1🟡. The other player bets the **Big Blind** of 2🟡.
    
2. Each player is delt a face-down card that only they can see. 
    
3. Now, the (first and only) betting round starts. The Bet Size must be a power of 2, and no bigger than ==32 , the **Bet Limit**==. So, the SB may: 
	- Fold (f), 
	- Call (c, 2🟡), 
	- Raise (R, 4🟡), 
	- Double Raise (D, 8🟡), 
	- Triple Raise (T, 16🟡) or 
	- Quadruple Raise (Q, 32🟡).
    
4. If one player fords, the other wins the pot. Otherwise, when they both players acted, and both bets are the same amount, we proceed to **Showdown**. First to show is the BB. [^1]
    
5. The highest card wins the pot! [^2]
     
6. Move the Button to the other player, and repeat!
    

---

## Examples

| Holes | Actions | Result       |
| ----- | ------- | ------------ |
| 7, 9  | f       | BB wins 1🟡  |
| 9, 8  | cc      | SB wins 2🟡  |
| 9, T  | cRc     | BB wins 4🟡  |
| J, Q  | RRc     | BB wins 8🟡  |
| A, J  | RDRf    | SB wins 16🟡 |
| K, A  | DRRc    | BB wins 32🟡 |
| A, A  | DDc     | chop         |

---

#### Footnotes

[^1]: Players who decide not to show their hand won't win the pot.
[^2]: The suit doesn't matter. If they both have the same face value, the pot is chopped.
