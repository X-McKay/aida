"""Chat mode for AIDA CLI."""

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

from aida.cli.ui.console import get_console
from aida.core.agent import Agent, AgentCapability, AgentConfig
from aida.tools.base import get_tool_registry, initialize_default_tools

chat_app = typer.Typer(name="chat")
console = get_console()
logger = logging.getLogger(__name__)

# Command shortcuts mapping
COMMAND_SHORTCUTS = {
    "?": "help",
    "h": "help",
    "s": "status",
    "c": "clear",
    "q": "exit",
    "t": "tools",
    "hist": "history",
}


class ChatSession:
    """Enhanced chat session with improved UX."""

    def __init__(self):
        """Initialize a new chat session.

        Sets up the chat environment with:
        - Tool registry for available tools
        - Empty conversation history
        - Session tracking with unique ID based on timestamp
        - Multiline input support for complex queries
        """
        self.agent: Agent | None = None
        # LLM will be accessed through orchestrator
        self.tool_registry = get_tool_registry()
        self.conversation_history: list[dict[str, Any]] = []
        self.session_started = datetime.utcnow()
        self.running = True
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.context_window = []
        self.multiline_mode = False
        self.multiline_buffer = []

    async def start(self):
        """Start chat session."""
        console.clear()
        self._show_welcome()

        # Initialize agent
        await self._setup_agent()

        # Main chat loop
        while self.running:
            try:
                await self._handle_user_input()
            except KeyboardInterrupt:
                if self.multiline_mode:
                    # Exit multiline mode
                    self.multiline_mode = False
                    self.multiline_buffer = []
                    console.print("[dim]Multiline input cancelled[/dim]")
                else:
                    # Confirm exit
                    if console.confirm("\nExit AIDA chat?", default=True):
                        break
                    console.print()
            except EOFError:
                break

        await self._cleanup()
        console.print("\n[dim]Chat session ended. Goodbye! 👋[/dim]")

    def _show_welcome(self):
        """Show enhanced welcome message."""
        welcome_content = [
            "[bold blue]Welcome to AIDA Chat![/bold blue]\n",
            "[yellow]Quick Commands:[/yellow]",
            "• [cyan]?[/cyan] or [cyan]/help[/cyan] - Show help",
            "• [cyan]@tool_name[/cyan] - Execute a tool directly",
            "• [cyan]#context[/cyan] - Manage conversation context",
            "• [cyan].exit[/cyan] - Exit chat\n",
            "[dim]Type your message or use commands. Press Ctrl+D for multiline input.[/dim]",
        ]

        console.print(
            Panel("\n".join(welcome_content), title="AIDA Chat v2.0", border_style="blue")
        )

    async def _setup_agent(self):
        """Setup chat agent with enhanced capabilities."""
        # Verify LLM is available
        try:
            from aida.llm import get_llm

            manager = get_llm()
            purposes = manager.list_purposes()
            if purposes:
                console.print(
                    f"[dim]✅ LLM configured with {len(purposes)} purposes (using Ollama)[/dim]"
                )
            else:
                console.print("[warning]⚠️  No LLM purposes available[/warning]")
        except Exception as e:
            console.print(f"[warning]⚠️  LLM check failed: {e}[/warning]")
            console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")

        # Initialize tools
        try:
            await initialize_default_tools()
            console.print("[dim]✅ Tools initialized[/dim]")
        except Exception as e:
            console.print(f"[warning]⚠️  Tool initialization warning: {e}[/warning]")

        # Create enhanced agent
        config = AgentConfig(
            name="aida_chat_agent",
            description="Enhanced chat agent with streaming and context management",
            capabilities=[
                AgentCapability(
                    name="chat_conversation",
                    description="Natural conversation with context awareness",
                ),
                AgentCapability(
                    name="tool_orchestration",
                    description="Intelligent tool selection and execution",
                ),
                AgentCapability(
                    name="llm_reasoning", description="Advanced reasoning with streaming responses"
                ),
            ],
        )

        self.agent = Agent(config)
        await self.agent.start()

        console.print("[dim]✅ Chat agent ready[/dim]\n")

    async def _handle_user_input(self):
        """Handle enhanced user input with multiline support."""
        # Build prompt with context indicator
        prompt_parts = ["[bold cyan]aida"]
        if self.context_window:
            prompt_parts.append(f"[dim]({len(self.context_window)} ctx)[/dim]")
        prompt_parts.append(">[/bold cyan]")
        prompt = "".join(prompt_parts)

        if self.multiline_mode:
            # Continue multiline input
            line = console.prompt("[dim]...[/dim]")
            if line is None:  # Ctrl+D pressed
                # Process multiline buffer
                user_input = "\n".join(self.multiline_buffer)
                self.multiline_mode = False
                self.multiline_buffer = []
                if user_input.strip():
                    await self._process_input(user_input)
            else:
                self.multiline_buffer.append(line)
        else:
            # Single line input
            try:
                user_input = console.prompt(prompt)
            except EOFError:
                # Start multiline mode
                self.multiline_mode = True
                console.print("[dim]Multiline mode - press Ctrl+D to send[/dim]")
                return

            if user_input is None:
                self.running = False
                return

            if not user_input.strip():
                return

            await self._process_input(user_input)

    async def _process_input(self, user_input: str):
        """Process user input with enhanced command handling."""
        # Check for quick exit
        if user_input.lower() in [".exit", ".quit", ".q"]:
            self.running = False
            return

        # Check for command shortcuts
        if user_input.startswith("/"):
            await self._handle_command(user_input)
        elif user_input.startswith("?"):
            # Quick help
            await self._handle_command("/help")
        elif user_input.startswith("@"):
            # Direct tool execution
            await self._handle_tool_shortcut(user_input)
        elif user_input.startswith("#"):
            # Context management
            await self._handle_context_command(user_input)
        else:
            # Regular chat message
            await self._handle_chat_message(user_input)

    async def _handle_command(self, command: str):
        """Handle slash commands with shortcuts."""
        parts = command.split()
        cmd = parts[0][1:].lower()  # Remove slash
        args = parts[1:] if len(parts) > 1 else []

        # Resolve shortcuts
        cmd = COMMAND_SHORTCUTS.get(cmd, cmd)

        command_handlers = {
            "help": self._show_help,
            "status": self._show_status,
            "tools": self._show_tools,
            "history": self._show_history,
            "clear": lambda: console.clear(),
            "exit": lambda: setattr(self, "running", False),
            "session": self._show_session_info,
            "save": self._save_session,
            "load": lambda: self._load_session(args[0] if args else None),
        }

        handler = command_handlers.get(cmd)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()
        else:
            console.print(f"[warning]Unknown command: /{cmd}[/warning]")
            console.print("Type [cyan]/help[/cyan] for available commands")

    async def _handle_tool_shortcut(self, input_str: str):
        """Handle @tool shortcuts for direct tool execution."""
        parts = input_str[1:].split(None, 1)
        if not parts:
            console.print("[warning]Usage: @tool_name [parameters][/warning]")
            return

        tool_name = parts[0]
        params_str = parts[1] if len(parts) > 1 else ""

        # Get tool
        tool = await self.tool_registry.get_tool(tool_name)
        if not tool:
            console.print(f"[error]Tool '{tool_name}' not found[/error]")
            tools = await self.tool_registry.list_tools()
            if tools:
                console.print(f"[dim]Available tools: {', '.join(tools)}[/dim]")
            return

        # Parse parameters (simple key=value format)
        params = {}
        if params_str:
            for pair in params_str.split():
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    params[key] = value

        try:
            console.print(f"[yellow]Executing {tool_name}...[/yellow]")
            result = await tool.execute_async(**params)

            # Format result
            if result.status.value == "completed":
                console.print("[success]✅ Success[/success]")
                if isinstance(result.result, dict):
                    console.print_json(result.result)
                else:
                    console.print(result.result)
            else:
                console.print(f"[error]❌ Failed: {result.error}[/error]")

        except Exception as e:
            console.print(f"[error]Tool execution error: {e}[/error]")

    async def _handle_context_command(self, input_str: str):
        """Handle context management commands."""
        parts = input_str[1:].split()
        if not parts:
            self._show_context_help()
            return

        cmd = parts[0].lower()

        if cmd == "show":
            self._show_context()
        elif cmd == "clear":
            self.context_window = []
            console.print("[dim]Context cleared[/dim]")
        elif cmd == "add":
            if len(parts) > 1:
                context_item = " ".join(parts[1:])
                self.context_window.append(context_item)
                console.print(f"[dim]Added to context: {context_item}[/dim]")
            else:
                console.print("[warning]Usage: #add <context item>[/warning]")
        elif cmd == "remove":
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < len(self.context_window):
                    removed = self.context_window.pop(idx)
                    console.print(f"[dim]Removed from context: {removed}[/dim]")
                else:
                    console.print("[error]Invalid context index[/error]")
            else:
                console.print("[warning]Usage: #remove <index>[/warning]")
        else:
            self._show_context_help()

    def _show_context_help(self):
        """Show context command help."""
        help_text = """[yellow]Context Commands:[/yellow]
• [cyan]#show[/cyan] - Show current context
• [cyan]#clear[/cyan] - Clear all context
• [cyan]#add <item>[/cyan] - Add item to context
• [cyan]#remove <index>[/cyan] - Remove item by index"""
        console.print(Panel(help_text, title="Context Help", border_style="yellow"))

    def _show_context(self):
        """Display current context window."""
        if not self.context_window:
            console.print("[dim]No context items[/dim]")
            return

        table = Table(title="Current Context")
        table.add_column("#", style="dim")
        table.add_column("Context Item")

        for idx, item in enumerate(self.context_window):
            table.add_row(str(idx), item)

        console.print(table)

    async def _handle_chat_message(self, message: str):
        """Handle chat message with streaming response."""
        # Add to history
        self.conversation_history.append(
            {"role": "user", "content": message, "timestamp": datetime.utcnow().isoformat()}
        )

        # Create progress indicator
        progress = Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        )

        with progress:
            task = progress.add_task("Planning...", total=None)

            try:
                # Import orchestrator
                from aida.core.orchestrator import get_orchestrator

                orchestrator = get_orchestrator()

                # Build context
                context = {
                    "session_id": self.session_id,
                    "conversation_history": self.conversation_history[-10:],  # Last 10 messages
                    "context_window": self.context_window,
                }

                # Track progress
                step_descriptions = []

                async def progress_callback(workflow, current_step):
                    step_num = workflow.current_step + 1
                    total_steps = len(workflow.steps)

                    if current_step.status == "running":
                        desc = f"Step {step_num}/{total_steps}: {current_step.purpose[:50]}..."
                        progress.update(task, description=desc)
                        step_descriptions.append(current_step.purpose)
                    elif current_step.status == "completed":
                        progress.update(
                            task, description=f"Step {step_num}/{total_steps} completed"
                        )

                # Execute request
                result = await orchestrator.execute_request(
                    message, context=context, progress_callback=progress_callback
                )

            finally:
                progress.stop()

        # Display response
        if result["status"] == "completed":
            response = await self._format_response(result["workflow"], result["results"])

            # Add to history
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Display with markdown rendering
            if "```" in response:
                # Has code blocks - render as markdown
                console.print(Markdown(response))
            else:
                # Plain text response
                console.print(f"\n{response}")

            # Show execution summary for complex workflows
            summary = result.get("execution_summary", {})
            if summary.get("total_steps", 0) > 1:
                console.print(
                    f"\n[dim]✅ {summary['completed_steps']}/{summary['total_steps']} steps "
                    f"({summary['success_rate']:.0f}%) in {summary['total_execution_time']:.1f}s[/dim]"
                )
        else:
            error_msg = result.get("error", "Unknown error")
            console.print(f"[error]❌ Error: {error_msg}[/error]")

    async def _format_response(self, workflow_data: dict, step_results: list) -> str:
        """Format workflow response for display."""
        response_parts = []

        # Extract key results
        for result in step_results:
            tool_name = result["step"]["tool_name"]

            if result.get("success") and result.get("result"):
                result_obj = result["result"]
                result_data = result_obj.get("result", {})

                # Check if the tool execution actually failed
                if result_obj.get("status") == "failed" and result_obj.get("error"):
                    response_parts.append(f"❌ {tool_name} failed: {result_obj['error']}")
                    continue

                if tool_name == "thinking" and isinstance(result_data, dict):
                    if "recommendations" in result_data:
                        recommendations = result_data["recommendations"]
                        if isinstance(recommendations, list):
                            response_parts.extend(recommendations)
                        else:
                            response_parts.append(str(recommendations))
                    elif "final_insight" in result_data:
                        response_parts.append(result_data["final_insight"])

                elif tool_name == "file_operations" and isinstance(result_data, dict):
                    step_params = result["step"].get("parameters", {})
                    operation = step_params.get("operation", "")

                    if operation == "write_file":
                        content = step_params.get("content", "")
                        path = result_data.get("path", step_params.get("path", ""))
                        response_parts.append(f"Created {path}:\n```\n{content}\n```")
                    elif operation == "list_files" and "files" in result_data:
                        files = result_data["files"]
                        if isinstance(files, list):
                            response_parts.append(
                                f"Found {len(files)} files:\n"
                                + "\n".join(f"- {f}" for f in files[:20])
                            )
                            if len(files) > 20:
                                response_parts.append(f"... and {len(files) - 20} more")
                    elif operation == "read_file" and "content" in result_data:
                        response_parts.append(
                            f"File content:\n```\n{result_data['content'][:500]}\n```"
                        )
                        if len(result_data["content"]) > 500:
                            response_parts.append("... (truncated)")
                    elif "path" in result_data:
                        response_parts.append(f"Operation completed on: {result_data['path']}")

                elif tool_name == "execution":
                    if (
                        isinstance(result_data, dict)
                        and "stdout" in result_data
                        and result_data["stdout"]
                    ):
                        output = result_data["stdout"].strip()
                        response_parts.append(f"Output:\n```\n{output}\n```")
                    else:
                        # Debug: show what we got
                        logger.debug(f"Execution result_data: {result_data}")

                elif tool_name == "system" and isinstance(result_data, dict):
                    if "stdout" in result_data and result_data["stdout"]:
                        response_parts.append(
                            f"System output:\n```\n{result_data['stdout'].strip()}\n```"
                        )
                    elif "output" in result_data:
                        response_parts.append(
                            f"System output:\n```\n{result_data['output'].strip()}\n```"
                        )
                    if "stderr" in result_data and result_data["stderr"]:
                        response_parts.append(
                            f"Error output:\n```\n{result_data['stderr'].strip()}\n```"
                        )

                elif tool_name == "llm_response":
                    # For llm_response tool, the result is the direct string response
                    if isinstance(result_data, str):
                        response_parts.append(result_data)
                    else:
                        # If it's wrapped in a dict or something else, extract the content
                        logger.debug(
                            f"llm_response result_data type: {type(result_data)}, value: {result_data}"
                        )
                        response_parts.append(str(result_data))

                elif tool_name == "thinking" and isinstance(result_data, dict):
                    # Handle thinking tool output
                    if "analysis" in result_data:
                        response_parts.append(result_data["analysis"])
                    elif "summary" in result_data:
                        response_parts.append(result_data["summary"])
                    else:
                        # Show structured data
                        response_parts.append(json.dumps(result_data, indent=2))

                else:
                    # Catch-all for any other tools
                    logger.debug(f"Unhandled tool '{tool_name}' with result: {result_data}")
                    if isinstance(result_data, str) and result_data.strip():
                        response_parts.append(result_data)
                    elif isinstance(result_data, dict) and result_data:
                        # Try to extract meaningful content from dict
                        if "output" in result_data:
                            response_parts.append(str(result_data["output"]))
                        elif "result" in result_data:
                            response_parts.append(str(result_data["result"]))
                        else:
                            # Show the whole dict in a readable format
                            response_parts.append(json.dumps(result_data, indent=2))

        if not response_parts:
            response_parts.append("Task completed successfully.")

        return "\n\n".join(response_parts)

    def _show_help(self):
        """Show enhanced help information."""
        help_sections = [
            (
                "Commands",
                [
                    ("/help, /?", "Show this help"),
                    ("/status, /s", "Show system status"),
                    ("/tools, /t", "List available tools"),
                    ("/history", "Show conversation history"),
                    ("/session", "Show session info"),
                    ("/save [name]", "Save session"),
                    ("/load <name>", "Load session"),
                    ("/clear", "Clear screen"),
                    ("/exit, .exit", "Exit chat"),
                ],
            ),
            (
                "Shortcuts",
                [
                    ("@tool_name", "Execute tool directly"),
                    ("#show", "Show context window"),
                    ("#add <item>", "Add to context"),
                    ("#clear", "Clear context"),
                    ("Ctrl+D", "Toggle multiline input"),
                ],
            ),
            (
                "Examples",
                [
                    ("@file_operations operation=list path=.", "List files"),
                    ('@thinking query="How to optimize this?"', "Get analysis"),
                    ("#add Project: AI Assistant", "Add context"),
                    ("Write a Python hello world script", "Natural request"),
                ],
            ),
        ]

        for section_title, items in help_sections:
            table = Table(title=section_title, show_header=False)
            table.add_column("Command", style="cyan")
            table.add_column("Description")

            for cmd, desc in items:
                table.add_row(cmd, desc)

            console.print(table)
            console.print()

    async def _show_status(self):
        """Show enhanced system status."""
        table = Table(title="AIDA Chat Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        # Session info
        duration = datetime.utcnow() - self.session_started
        table.add_row("Session", "Active", f"ID: {self.session_id}, Duration: {duration}")

        # Agent status
        if self.agent:
            table.add_row(
                "Agent", "Running" if self.agent._started else "Stopped", f"Name: {self.agent.name}"
            )

        # Tools
        tools_count = len(await self.tool_registry.list_tools())
        table.add_row("Tools", "Loaded", f"{tools_count} tools available")

        # LLM
        try:
            from aida.llm import get_llm

            manager = get_llm()
            purposes = manager.list_purposes()
            table.add_row("LLM", "Ready", f"{len(purposes)} purposes (Ollama)")
        except Exception:
            table.add_row("LLM", "Not Ready", "Ollama not running")

        # Context
        table.add_row(
            "Context Window",
            "Active" if self.context_window else "Empty",
            f"{len(self.context_window)} items",
        )

        # History
        table.add_row("Conversation", "Recording", f"{len(self.conversation_history)} messages")

        console.print(table)

    async def _show_tools(self):
        """Show available tools with descriptions."""
        tools = await self.tool_registry.list_tools()

        if not tools:
            console.print("[warning]No tools available[/warning]")
            return

        table = Table(title="Available Tools")
        table.add_column("Tool", style="tool")
        table.add_column("Shortcut", style="cyan")
        table.add_column("Description")

        for tool_name in sorted(tools):
            tool = await self.tool_registry.get_tool(tool_name)
            if tool:
                shortcut = f"@{tool_name}"
                table.add_row(tool_name, shortcut, tool.description)

        console.print(table)

    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            console.print("[dim]No conversation history[/dim]")
            return

        console.print(
            f"[bold]Conversation History[/bold] ({len(self.conversation_history)} messages)\n"
        )

        # Show last 10 messages
        for msg in self.conversation_history[-10:]:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")

            # Format message
            if role == "user":
                console.print(f"[cyan]You:[/cyan] {content}")
            else:
                console.print(f"[blue]AIDA:[/blue] {content}")

            if timestamp:
                dt = datetime.fromisoformat(timestamp)
                console.print(f"[dim]{dt.strftime('%H:%M:%S')}[/dim]\n")

    def _show_session_info(self):
        """Show detailed session information."""
        duration = datetime.utcnow() - self.session_started

        info = {
            "session_id": self.session_id,
            "started": self.session_started.isoformat(),
            "duration": str(duration),
            "messages": len(self.conversation_history),
            "context_items": len(self.context_window),
            "multiline_mode": self.multiline_mode,
        }

        console.print(
            Panel(json.dumps(info, indent=2), title="Session Information", border_style="blue")
        )

    async def _save_session(self, name: str | None = None):
        """Save current session."""
        if not name:
            name = self.session_id

        session_dir = Path.home() / ".aida" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

        session_file = session_dir / f"{name}.json"

        session_data = {
            "session_id": self.session_id,
            "started": self.session_started.isoformat(),
            "conversation_history": self.conversation_history,
            "context_window": self.context_window,
            "saved_at": datetime.utcnow().isoformat(),
        }

        try:
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            console.print(f"[success]✅ Session saved to: {session_file}[/success]")
        except Exception as e:
            console.print(f"[error]Failed to save session: {e}[/error]")

    async def _load_session(self, name: str):
        """Load a saved session."""
        if not name:
            console.print("[warning]Usage: /load <session_name>[/warning]")
            return

        session_dir = Path.home() / ".aida" / "sessions"
        session_file = session_dir / f"{name}.json"

        if not session_file.exists():
            console.print(f"[error]Session not found: {name}[/error]")
            # List available sessions
            sessions = list(session_dir.glob("*.json"))
            if sessions:
                console.print("[dim]Available sessions:[/dim]")
                for s in sessions:
                    console.print(f"  - {s.stem}")
            return

        try:
            with open(session_file) as f:
                session_data = json.load(f)

            self.conversation_history = session_data.get("conversation_history", [])
            self.context_window = session_data.get("context_window", [])

            console.print(f"[success]✅ Session loaded: {name}[/success]")
            console.print(
                f"[dim]Restored {len(self.conversation_history)} messages and {len(self.context_window)} context items[/dim]"
            )

        except Exception as e:
            console.print(f"[error]Failed to load session: {e}[/error]")

    async def _cleanup(self):
        """Cleanup session resources."""
        if self.agent:
            await self.agent.stop()

        # Auto-save session
        await self._save_session()

        console.print("[dim]Session saved automatically[/dim]")


@chat_app.command()
def start():
    """Start AIDA chat mode."""

    async def _start():
        session = ChatSession()
        await session.start()

    try:
        asyncio.run(_start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session interrupted[/yellow]")
    except Exception as e:
        console.print(f"[error]Chat mode failed: {e}[/error]")


# Default command when called without subcommand
@chat_app.callback(invoke_without_command=True)
def chat_main(ctx: typer.Context):
    """Start AIDA chat mode."""
    if ctx.invoked_subcommand is None:
        start()
