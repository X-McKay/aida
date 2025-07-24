# AIDA Text User Interface (TUI)

The AIDA TUI provides a terminal-based interface for interacting with the AIDA system, featuring real-time monitoring and an enhanced chat experience.

## Features

### Layout

The TUI is divided into two main sections:

1. **Left Panel (2/3 width)**: Chat Interface
   - Interactive chat with AIDA
   - Message history with role indicators (USER/AGENT/SYSTEM)
   - Real-time response streaming
   - Command support

2. **Right Panel (1/3 width)**: Monitoring Dashboard
   - **Resource Monitor (top)**: Live charts showing CPU, Memory, and GPU usage
   - **Ongoing Tasks (middle)**: Active task tracking with status indicators
   - **Available Agents (bottom)**: List of registered agents and their status

### Key Features

- **Real-time Updates**: All panels update automatically
- **Keyboard Shortcuts**:
  - `Ctrl+C`: Quit the application
  - `Ctrl+L`: Clear chat history
  - `Ctrl+T`: Toggle between light/dark themes
- **Responsive Design**: Adapts to terminal size changes
- **Task Tracking**: Monitor ongoing operations
- **Agent Status**: See which agents are available and their current state

## Usage

### Starting the TUI

```bash
# Using uv
uv run aida tui

# Or with Python
python -m aida.cli.main tui
```

### Chat Interface

Type your messages in the input field at the bottom of the chat panel. The TUI supports:

- Natural language queries
- Multi-line input (press Enter to send)
- Context-aware conversations
- Tool execution feedback

### Resource Monitoring

The resource monitor displays:
- **CPU**: Real-time CPU usage percentage
- **Memory**: System memory utilization
- **GPU**: GPU usage (if available, otherwise simulated)

Charts use ASCII characters to show trends over time:
- Green lines: Low usage (0-40%)
- Yellow lines: Medium usage (40-80%)
- Red lines: High usage (80-100%)

### Task Management

The tasks panel shows:
- Task ID and description
- Status indicators:
  - `○` Pending (yellow)
  - `●` Running (green)
  - `✓` Completed (dim green)

### Agent Monitoring

The agents panel displays:
- Agent type and ID
- Status indicators:
  - `●` Ready (green)
  - `○` Busy (yellow)
  - `×` Offline (red)

## Requirements

- Terminal with UTF-8 support
- Minimum terminal size: 80x24 characters
- Python 3.11 or higher
- Textual 0.47.0 or higher

## Troubleshooting

### TUI doesn't start
- Ensure Textual is installed: `pip install textual>=0.47.0`
- Check terminal compatibility: The TUI requires a modern terminal emulator

### Characters appear garbled
- Ensure your terminal supports UTF-8 encoding
- Try setting: `export LANG=en_US.UTF-8`

### Performance issues
- Large terminal sizes may impact performance
- Consider reducing update intervals in resource monitoring
- Close other resource-intensive applications

## Development

### Extending the TUI

To add new widgets or features:

1. Create a new widget in `aida/cli/tui/widgets/`
2. Import and add to the layout in `aida/cli/tui/app.py`
3. Follow the Textual widget pattern for reactive updates

### Custom Themes

Modify the CSS in `app.py` to customize colors and styling:

```python
CSS = """
Screen {
    background: $surface;
}
# Add your custom styles here
"""
```

## Future Enhancements

- [ ] Log viewer panel
- [ ] Configuration editor
- [ ] Task detail view
- [ ] Agent communication visualization
- [ ] Custom dashboard layouts
- [ ] Export conversation history
- [ ] Plugin system for custom widgets
