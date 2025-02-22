# Connect 4 Simulator

## Overview
This is a Python-based Connect 4 simulator, where players take turns dropping pieces into a 7-column game board. The game follows the standard Connect 4 rules, where the first player to connect four pieces in a row, column, or diagonal wins.

The program runs in the terminal, allowing users to input their moves interactively

## Features
- **Text-based board display** in the console.
- **Two-player gameplay** with user inputs.
- **Automatic game state updates** after each move.
- **Win detection** for horizontal, vertical, and diagonal connections.
- **Option to quit** by entering `-1`.
---
## Installation & Setup
### Prerequisites
Ensure you have Python3.x installed on your system. If you are using Anaconda, Python is already included.
### Running the Game
1. Clone this repository or download the project files.
2. Open a **terminal** or **Anaconda Prompt** and navigate to the project folder.
```
cd path/to/connectfour
```
3. Run the `simulator.py` script using Python:
```
phython simulator.py
```
4. Follow the on-screen instructions to play the game.
---
## How to Play
- The board starts empty:
```
[[0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0]  
 [0 0 0 0 0 0 0]]
```
- Players take turns to drop their pieces into one of the **7 columns (0-6)**.
- The game prompts the current player to **enter a column number** where they want to drop their piece.
- If a player enters `-1`, the game **ends immediately**.
- The program checks for a **winter after every move**. If a player connects four pieces, the game announces the winner.

### Example Gameplay
```
[Player 1] (-1 to quit) Drop Piece to: 3
[Player 2] (-1 to quit) Drop Piece to: 4
[Player 1] (-1 to quit) Drop Piece to: 3
```
- The board updates after each move.
- If `Actions: []` appears, it means no valid moves are available.
- When a player wins, the program prints:
```
Winner: Player 1
```

