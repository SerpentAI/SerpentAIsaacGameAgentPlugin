from serpent.utilities import clear_terminal


class TerminalPrinter:

    def __init__(self):
        self.lines = list()

    def add(self, content):
        self.lines.append(content)

    def empty_line(self):
        self.lines.append("")

    def clear(self):
        self.lines = list()

    def flush(self):
        clear_terminal()
        print("\n".join(self.lines))
        self.clear()
