"""Chat widget for AIDA TUI."""

import asyncio
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Static

from aida.core.orchestrator import get_orchestrator
from aida.llm import get_llm
from aida.tools.base import initialize_default_tools


class ChatMessage(Static):
    """A single chat message."""

    def __init__(self, role: str, content: str, timestamp: datetime | None = None):
        """Initialize a chat message."""
        self.role = role
        self.timestamp = timestamp or datetime.now()

        # Format message with role indicator
        if role == "user":
            formatted = Text()
            formatted.append("USER\n", style="bold cyan")
            formatted.append(content, style="white")
        elif role == "agent":
            formatted = Text()
            formatted.append("AGENT\n", style="bold green")
            formatted.append(content, style="white")
        elif role == "system":
            formatted = Text()
            formatted.append("SYSTEM\n", style="bold yellow")
            formatted.append(content, style="dim white")
        else:
            formatted = Text(content)

        super().__init__(formatted)


class ChatWidget(Widget):
    """Chat interface widget."""

    def __init__(self):
        """Initialize the chat widget."""
        super().__init__()
        self._last_input_time = 0
        self._input_debounce = 0.1  # 100ms debounce for input events

    CSS = """
    ChatWidget {
        height: 100%;
        layout: vertical;
    }

    #messages-container {
        height: 1fr;
        overflow-y: scroll;
        padding: 1;
        margin-bottom: 1;
    }

    #chat-input {
        height: 3;
        dock: bottom;
        margin: 0 1 1 1;
        border: solid $primary;
        background: $surface;
    }

    ChatMessage {
        margin-bottom: 1;
    }

    .system-message {
        color: $text-muted;
        text-style: italic;
    }
    """

    conversation_history: reactive[list[dict]] = reactive([])

    class MessageSent(Message):
        """Message sent event."""

        def __init__(self, content: str) -> None:
            """Initialize the message sent event."""
            self.content = content
            super().__init__()

    def compose(self) -> ComposeResult:
        """Create the chat UI."""
        with Vertical():
            # Chat messages container
            with ScrollableContainer(id="messages-container"):
                yield Static("Welcome to AIDA! Type your message below.", classes="system-message")

            # Input field with protection against automatic value changes
            input_widget = Input(
                placeholder="How can I assist you today?",
                id="chat-input",
                restrict=None,  # No input restrictions
                max_length=1000,  # Reasonable limit
            )
            # Ensure input starts empty
            input_widget.value = ""
            yield input_widget

    async def initialize(self) -> None:
        """Initialize the chat session."""
        # Defer initialization messages until widget is mounted
        messages = []

        # Initialize tools
        try:
            await initialize_default_tools()
            messages.append(("system", "Tools initialized successfully."))
        except Exception as e:
            messages.append(("system", f"Warning: Tool initialization failed: {e}"))

        # Check LLM
        try:
            manager = get_llm()
            purposes = manager.list_purposes()
            if purposes:
                messages.append(("system", f"LLM ready with {len(purposes)} purposes."))
            else:
                messages.append(("system", "Warning: No LLM purposes available."))
        except Exception as e:
            messages.append(("system", f"Warning: LLM check failed: {e}"))

        # Post messages after a short delay to ensure widget is mounted
        await asyncio.sleep(0.1)
        for role, msg in messages:
            self.add_message(role, msg)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        import re
        import time

        # Debounce rapid submissions
        current_time = time.time()
        if current_time - self._last_input_time < self._input_debounce:
            event.stop()
            return
        self._last_input_time = current_time

        # Clean the input value - remove any escape sequences that might have leaked in
        raw_value = event.value
        # Strip ANSI escape sequences
        clean_value = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", raw_value)
        # Strip cursor positioning sequences
        clean_value = re.sub(r"\x1b\[\d+;\d+[Hf]", "", clean_value)
        # Strip any other control characters except newlines and tabs
        clean_value = "".join(c for c in clean_value if c.isprintable() or c in "\n\t")

        message = clean_value.strip()
        if not message:
            return

        # Clear input immediately and prevent event propagation
        event.input.value = ""
        event.stop()

        # Add user message
        self.add_message("user", message)

        # Process message in background
        self.run_worker(self.process_message(message))

    async def process_message(self, message: str) -> None:
        """Process user message through orchestrator."""
        import io
        import sys

        try:
            # Get orchestrator
            orchestrator = get_orchestrator()

            # Create context
            context = {
                "session_id": f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "conversation_history": self.conversation_history[-10:],  # Last 10 messages
                "interface": "tui",
            }

            # Show thinking indicator
            self.post_system_message("Thinking...")

            # Capture any stdout/stderr output during orchestrator execution
            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr

            try:
                # Redirect stdout/stderr to capture any rogue output
                sys.stdout = captured_stdout
                sys.stderr = captured_stderr

                # Execute request
                result = await orchestrator.execute_request(message, context=context)
            finally:
                # Always restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Clear thinking message
            self.clear_last_system_message()

            # Display response
            if result["status"] == "completed":
                response = self._format_response(result)
                self.add_message("agent", response)
            else:
                error_msg = result.get("error", "Unknown error")
                self.add_message("system", f"Error: {error_msg}")

        except Exception as e:
            self.clear_last_system_message()
            self.add_message("system", f"Error: {str(e)}")

    def _format_response(self, result: dict) -> str:
        """Format orchestrator response for display."""
        # Handle new orchestrator response format
        if "result" in result and isinstance(result["result"], dict):
            execution_result = result["result"]

            # Check if we have step results
            if "step_results" in execution_result:
                response_parts = []
                for _step_id, step_result in execution_result["step_results"].items():
                    if step_result.get("status") == "completed" and "result" in step_result:
                        # Extract the actual result content
                        step_data = step_result["result"]
                        if isinstance(step_data, dict):
                            # Look for various result formats
                            if "content" in step_data:
                                response_parts.append(step_data["content"])
                            elif "text" in step_data:
                                response_parts.append(step_data["text"])
                            elif "code" in step_data:
                                response_parts.append(f"```python\n{step_data['code']}\n```")
                            elif "output" in step_data:
                                response_parts.append(step_data["output"])
                            else:
                                # Try to format the dict nicely
                                response_parts.append(str(step_data))
                        else:
                            response_parts.append(str(step_data))

                if response_parts:
                    return "\n\n".join(response_parts)

            # Check for summary
            if "summary" in execution_result:
                return execution_result["summary"]

        # Handle old format (backward compatibility)
        if "workflow" in result and "results" in result:
            # Format based on the type of operation
            step_results = result.get("results", [])
            response_parts = []

            for step_result in step_results:
                if step_result.get("success"):
                    tool_name = step_result.get("step", {}).get("tool_name", "")
                    result_data = step_result.get("result", {})

                    if tool_name == "llm_response":
                        # Direct LLM response
                        if isinstance(result_data, dict) and "result" in result_data:
                            response_parts.append(result_data["result"])
                        else:
                            response_parts.append(str(result_data))
                    elif (
                        tool_name == "thinking"
                        and isinstance(result_data, dict)
                        and "result" in result_data
                    ):
                        # Analysis results
                        analysis = result_data["result"]
                        if isinstance(analysis, dict) and "analysis" in analysis:
                            response_parts.append(analysis["analysis"])

            return "\n\n".join(response_parts) if response_parts else "Task completed successfully."

        # Fallback to simple message
        return result.get("message", "Task completed.")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat."""
        # Update history
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

        # Add to UI
        container = self.query_one("#messages-container")
        message_widget = ChatMessage(role, content)
        container.mount(message_widget)

        # Scroll to bottom
        container.scroll_end(animate=True)

    def post_system_message(self, message: str) -> None:
        """Post a system message."""
        self.add_message("system", message)

    def clear_last_system_message(self) -> None:
        """Remove the last system message."""
        container = self.query_one("#messages-container")
        messages = list(container.query(ChatMessage))

        # Find and remove last system message
        for message in reversed(messages):
            if hasattr(message, "role") and message.role == "system":
                message.remove()
                break

    def clear_chat(self) -> None:
        """Clear all chat messages."""
        self.conversation_history.clear()
        container = self.query_one("#messages-container")

        # Remove all messages except welcome
        for message in container.query(ChatMessage):
            message.remove()

        self.post_system_message("Chat cleared. How can I assist you?")

    def on_focus(self) -> None:
        """Focus the input when widget gets focus."""
        self.query_one("#chat-input").focus()

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Monitor for unexpected input changes and clean escape sequences."""
        import re

        # Check for escape sequences or control characters
        if "\x1b" in event.value or any(ord(c) < 32 and c not in "\n\t" for c in event.value):
            # Clean the input immediately
            clean_value = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", event.value)
            clean_value = re.sub(r"\x1b\[\d+;\d+[Hf]", "", clean_value)
            clean_value = "".join(c for c in clean_value if c.isprintable() or c in "\n\t")

            # Update the input field with clean value
            event.input.value = clean_value
            event.stop()
            return

        # If the input has grown unexpectedly large, it might be the duplication bug
        if len(event.value) > 100 and event.value.count(event.value[0]) == len(event.value):
            # Clear the input if it looks like repeated characters
            event.input.value = ""
            event.stop()
            return

        # Additional check for rapid repeated characters
        if len(event.value) > 10:
            # Check if the last 10 characters are all the same
            last_chars = event.value[-10:]
            if len(set(last_chars)) == 1:
                # Likely a bug, truncate to reasonable length
                event.input.value = event.value[:-9]
                event.stop()
