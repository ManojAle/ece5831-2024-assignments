# game.py
import random
from draw_initialization import draw_game, clear_screen



def play_game():
    """
    Simulate the game logic. This function will return the game result,
    which could be 'win', 'lose', or 'draw'.
    """
    print("Playing game...")
    return "win"  # Simulating the game result, change to 'lose' or 'draw' if needed

def main():
    # Clear the screen before playing the game
    clear_screen()

    # Play the game and get the result
    result = play_game()

    # Pass the result to draw_game to display the result
    draw_game(result)

if __name__ == '__main__':
    main()

