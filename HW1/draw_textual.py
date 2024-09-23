# draw.py
import os

def draw_game(result):
    """
    This function takes the result of the game and displays it.
    For this example, we just print the result.
    """
    if result == "win":
        print("Congratulations! You won the game!")
    elif result == "lose":
        print("You lost the game. Better luck next time!")
    else:
        print("It's a draw!")

def clear_screen():
    """
    This function clears the screen using an OS command.
    On Windows, the 'cls' command is used, while on Unix-based
    systems, 'clear' is used.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
