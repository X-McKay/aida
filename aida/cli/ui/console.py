"""Console utilities for AIDA CLI."""

from rich.console import Console as RichConsole
from rich.theme import Theme
from rich.highlighter import ReprHighlighter
from rich.traceback import install
from rich.logging import RichHandler
import logging


# Custom theme for AIDA
aida_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold blue",
    "accent": "magenta",
    "dim": "dim white",
    "prompt": "bold cyan",
    "agent": "bold blue",
    "tool": "bold yellow",
    "provider": "bold green",
    "status.running": "green",
    "status.stopped": "red",
    "status.error": "bold red",
    "status.pending": "yellow",
})


class Console:
    """Enhanced console for AIDA CLI."""
    
    def __init__(self):
        self.console = RichConsole(
            theme=aida_theme,
            highlighter=ReprHighlighter()
        )
    
    def print(self, *args, **kwargs):
        """Print with rich formatting."""
        self.console.print(*args, **kwargs)
    
    def print_json(self, data, indent=2):
        """Print JSON data with syntax highlighting."""
        from rich.syntax import Syntax
        import json
        
        json_str = json.dumps(data, indent=indent, default=str)
        syntax = Syntax(json_str, "json", theme="monokai")
        self.console.print(syntax)
    
    def print_yaml(self, data):
        """Print YAML data with syntax highlighting."""
        from rich.syntax import Syntax
        import yaml
        
        yaml_str = yaml.dump(data, default_flow_style=False)
        syntax = Syntax(yaml_str, "yaml", theme="monokai")
        self.console.print(syntax)
    
    def print_code(self, code, language="python"):
        """Print code with syntax highlighting."""
        from rich.syntax import Syntax
        
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)
    
    def print_table(self, table):
        """Print a rich table."""
        self.console.print(table)
    
    def print_panel(self, content, title=None, **kwargs):
        """Print content in a panel."""
        from rich.panel import Panel
        
        panel = Panel(content, title=title, **kwargs)
        self.console.print(panel)
    
    def status(self, message):
        """Create a status context manager."""
        return self.console.status(message)
    
    def prompt(self, message, default=None, password=False):
        """Prompt for user input."""
        if password:
            import getpass
            try:
                # Use rich.prompt for proper rendering
                from rich.prompt import Prompt
                return Prompt.ask(message, password=True, console=self.console)
            except EOFError:
                return None
            except KeyboardInterrupt:
                self.print("\n[warning]Cancelled by user[/warning]")
                return None
        else:
            try:
                # Use rich.prompt for proper markup rendering
                from rich.prompt import Prompt
                if default:
                    return Prompt.ask(message, default=default, console=self.console)
                else:
                    return Prompt.ask(message, console=self.console)
            except EOFError:
                # Handle EOF gracefully (stdin closed)
                return None
            except KeyboardInterrupt:
                self.print("\n[warning]Cancelled by user[/warning]")
                return None
    
    def confirm(self, message, default=True):
        """Prompt for yes/no confirmation."""
        default_str = "Y/n" if default else "y/N"
        
        while True:
            try:
                response = input(f"{message} [{default_str}]: ").strip().lower()
                
                if not response:
                    return default
                elif response in ['y', 'yes', 'true', '1']:
                    return True
                elif response in ['n', 'no', 'false', '0']:
                    return False
                else:
                    self.print("[warning]Please answer 'y' or 'n'[/warning]")
                    
            except KeyboardInterrupt:
                self.print("\n[warning]Cancelled by user[/warning]")
                return False
    
    def select(self, message, choices, default=None):
        """Prompt for selection from choices."""
        self.print(f"\n[prompt]{message}[/prompt]")
        
        for i, choice in enumerate(choices, 1):
            marker = " [dim](default)[/dim]" if choice == default else ""
            self.print(f"  {i}. {choice}{marker}")
        
        while True:
            try:
                response = input(f"\nSelect [1-{len(choices)}]: ").strip()
                
                if not response and default:
                    return default
                
                try:
                    index = int(response) - 1
                    if 0 <= index < len(choices):
                        return choices[index]
                    else:
                        self.print(f"[warning]Please select a number between 1 and {len(choices)}[/warning]")
                except ValueError:
                    self.print("[warning]Please enter a valid number[/warning]")
                    
            except KeyboardInterrupt:
                self.print("\n[warning]Cancelled by user[/warning]")
                return None
    
    def clear(self):
        """Clear the console."""
        self.console.clear()
    
    def rule(self, title=None, style="dim"):
        """Print a horizontal rule."""
        from rich.rule import Rule
        self.console.print(Rule(title, style=style))
    
    def progress(self):
        """Create a progress context manager."""
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn, BarColumn, 
            TaskProgressColumn, TimeElapsedColumn
        )
        
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
    
    def live(self, renderable):
        """Create a live display context manager."""
        from rich.live import Live
        return Live(renderable, console=self.console)
    
    def pager(self, content):
        """Display content in a pager."""
        with self.console.pager():
            self.console.print(content)


def setup_console():
    """Setup and configure the global console."""
    # Install rich tracebacks
    install(console=RichConsole(theme=aida_theme))
    
    # Setup rich logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=RichConsole(theme=aida_theme))]
    )
    
    return Console()


# Global console instance
_console = None


def get_console():
    """Get the global console instance."""
    global _console
    if _console is None:
        _console = setup_console()
    return _console