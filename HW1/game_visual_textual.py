# game.py
import random
visual_mode = random.choice([1,None])
if visual_mode:
    # in visual mode, we draw using graphics
    import draw_visual as draw
else:
    # in textual mode, we print out text
    import draw_textual as draw

def play_game():
    """
    Simulates playing a game by randomly selecting a result.
    The result can either be 'win', 'lose', or 'draw'.
    """
    print("Playing the game...")
    outcomes = ["win", "lose", "draw"]
    return random.choice(outcomes)

def main():
    """
    The main function that clears the screen, plays the game,
    and draws the result.
    """
    # Clear the screen first
    draw.clear_screen()

    # Play the game and get the result
    result = play_game()

    # Draw the game result
    draw.draw_game(result)

if __name__ == '__main__':
    main()
