# draw.py
import os

def draw_game(result):
    """
    This function takes the result of the game and displays it with
    some visual ASCII art based on the result.
    """
    if result == "win":
        print("🎉 Congratulations! You Won! 🎉")
        print(r"""
 __     ______  _    _  __          _______ _   _ 
 \ \   / / __ \| |  | | \ \        / /_   _| \ | |
  \ \_/ / |  | | |  | |  \ \  /\  / /  | | |  \| |
   \   /| |  | | |  | |   \ \/  \/ /   | | | . ` |
    | | | |__| | |__| |    \  /\  /   _| |_| |\  |
    |_|  \____/ \____/      \/  \/   |_____|_| \_|
        """)
    elif result == "lose":
        print("💔 You lost the game. Better luck next time!")
        print(r"""
  __     ______  _    _   _      ____   _____ ______ 
  \ \   / / __ \| |  | | | |    / __ \ / ____|  ____|
   \ \_/ / |  | | |  | | | |   | |  | | (___ | |__   
    \   /| |  | | |  | | | |   | |  | |\___ \|  __|  
     | | | |__| | |__| | | |___| |__| |____) | |____ 
     |_|  \____/ \____/  |______\____/|_____/|______|
        """)
    else:
        print("🤝 It's a draw! 🤝")
        

def clear_screen():
    """
    This function clears the screen using an OS command.
    On Windows, the 'cls' command is used, while on Unix-based
    systems, 'clear' is used.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
