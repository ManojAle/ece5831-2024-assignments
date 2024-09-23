# draw.py
class Screen:
    def __init__(self):
        self.contents = []

    def add_content(self, content):
        self.contents.append(content)

    def clear(self):
        self.contents = []
        print("Screen cleared!")

    def display(self):
        for content in self.contents:
            print(content)

def draw_game(result):
    """
    This function takes the game result and displays it using
    the main_screen singleton object initialized in this module.
    """
    main_screen.add_content(f"🎮 Game result: {result}")
    main_screen.display()

def clear_screen():
    """
    This function clears the main_screen using the singleton object.
    """
    main_screen.clear()

# Initialize the screen as a singleton
main_screen = Screen()
