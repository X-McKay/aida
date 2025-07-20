"""Interactive mode for AIDA CLI."""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from aida.cli.ui.console import get_console
from aida.core.agent import Agent, AgentConfig, AgentCapability
from aida.llm import get_llm
from aida.tools.base import get_tool_registry, initialize_default_tools


interactive_app = typer.Typer(name="interactive")
console = get_console()


class InteractiveSession:
    """Interactive AIDA session."""
    
    def __init__(self):
        self.agent: Optional[Agent] = None
        self.llm_manager = get_llm()
        self.tool_registry = get_tool_registry()
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_started = datetime.utcnow()
        self.running = True
    
    async def start(self):
        """Start interactive session."""
        console.clear()
        self._show_welcome()
        
        # Initialize basic agent
        await self._setup_default_agent()
        
        # Main interaction loop
        while self.running:
            try:
                await self._handle_user_input()
            except KeyboardInterrupt:
                if console.confirm("\nExit AIDA interactive mode?", default=True):
                    break
                else:
                    console.print()
            except EOFError:
                break
        
        await self._cleanup()
        console.print("\n[dim]Goodbye! 👋[/dim]")
    
    def _show_welcome(self):
        """Show welcome message."""
        welcome_text = """[bold blue]Welcome to AIDA Interactive Mode![/bold blue]

[yellow]Available commands:[/yellow]
• [cyan]/help[/cyan] - Show help
• [cyan]/status[/cyan] - Show system status  
• [cyan]/agents[/cyan] - Manage agents
• [cyan]/tools[/cyan] - List available tools
• [cyan]/llm[/cyan] - Manage LLM providers
• [cyan]/history[/cyan] - Show conversation history
• [cyan]/clear[/cyan] - Clear screen
• [cyan]/exit[/cyan] or [cyan]/quit[/cyan] - Exit

[dim]Type your message or use commands. Press Ctrl+C twice to exit.[/dim]"""
        
        console.print_panel(welcome_text, title="AIDA Interactive", border_style="blue")
    
    async def _setup_default_agent(self):
        """Setup a default agent for the session."""
        # Initialize LLM providers first
        try:
            from aida.config.llm_defaults import auto_configure_llm_providers
            await auto_configure_llm_providers()
            console.print("[dim]✅ LLM providers configured[/dim]")
        except Exception as e:
            console.print(f"[warning]⚠️  LLM provider setup failed: {e}[/warning]")
            console.print("[dim]Will use fallback planning without LLM[/dim]")
        
        # Initialize tools
        try:
            await initialize_default_tools()
            console.print("[dim]✅ Tools initialized[/dim]")
        except Exception as e:
            console.print(f"[warning]⚠️  Tool initialization failed: {e}[/warning]")
        
        config = AgentConfig(
            name="interactive_agent",
            description="Default agent for interactive mode with LLM reasoning",
            capabilities=[
                AgentCapability(
                    name="interactive_chat",
                    description="Interactive conversation with workflow orchestration"
                ),
                AgentCapability(
                    name="tool_orchestration", 
                    description="Automatic tool selection and execution"
                ),
                AgentCapability(
                    name="llm_reasoning",
                    description="LLM-powered analysis and planning"
                )
            ]
        )
        
        self.agent = Agent(config)
        await self.agent.start()
        
        console.print("[dim]✅ Interactive agent with LLM reasoning initialized[/dim]\n")
    
    async def _handle_user_input(self):
        """Handle user input."""
        try:
            user_input = console.prompt("[bold cyan]aida>[/bold cyan]")
            
            # Handle EOF (None response)
            if user_input is None:
                console.print("\n[dim]Session ended[/dim]")
                self.running = False
                return
            
            # Handle empty input
            if not user_input.strip():
                return
            
            # Handle commands
            if user_input.startswith('/'):
                await self._handle_command(user_input)
            else:
                await self._handle_message(user_input)
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")
    
    async def _handle_command(self, command: str):
        """Handle interactive commands."""
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in ['/help', '/h']:
            self._show_help()
        elif cmd in ['/status', '/s']:
            await self._show_status()
        elif cmd in ['/agents', '/a']:
            await self._manage_agents(args)
        elif cmd in ['/tools', '/t']:
            await self._show_tools()
        elif cmd in ['/llm', '/l']:
            await self._manage_llm(args)
        elif cmd in ['/history', '/hist']:
            self._show_history()
        elif cmd in ['/clear', '/c']:
            console.clear()
        elif cmd in ['/exit', '/quit', '/q']:
            self.running = False
        elif cmd in ['/exec']:
            await self._execute_tool(args)
        elif cmd in ['/config']:
            await self._show_config()
        else:
            console.print(f"[warning]Unknown command: {cmd}[/warning]")
            console.print("Type [cyan]/help[/cyan] for available commands")
    
    async def _handle_message(self, message: str):
        """Handle chat message with LLM reasoning and tool orchestration."""
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        console.print(f"[dim]🧠 Planning workflow...[/dim]")
        
        try:
            # Import orchestrator
            from aida.core.orchestrator import get_orchestrator
            
            orchestrator = get_orchestrator()
            
            # Track step timing
            step_start_times = {}
            
            # Define progress callback
            async def progress_callback(workflow, current_step):
                step_num = workflow.current_step + 1
                total_steps = len(workflow.steps)
                
                # Track timing
                if current_step.status == "running":
                    step_start_times[step_num] = datetime.utcnow()
                    
                    # Show more details for different tool types
                    if current_step.tool_name == "thinking":
                        console.print(f"[dim]🤔 Step {step_num}/{total_steps}: Analyzing and planning approach...[/dim]")
                    elif current_step.tool_name == "file_operations":
                        op = current_step.parameters.get("operation", "")
                        if op == "write_file":
                            console.print(f"[dim]📝 Step {step_num}/{total_steps}: Creating script file...[/dim]")
                        else:
                            console.print(f"[dim]📁 Step {step_num}/{total_steps}: {current_step.purpose}[/dim]")
                    elif current_step.tool_name == "execution":
                        console.print(f"[dim]🚀 Step {step_num}/{total_steps}: Testing script execution...[/dim]")
                    else:
                        console.print(f"[dim]⚙️  Step {step_num}/{total_steps}: {current_step.purpose}[/dim]")
                
                elif current_step.status == "completed":
                    # Calculate execution time
                    if step_num in step_start_times:
                        elapsed = (datetime.utcnow() - step_start_times[step_num]).total_seconds()
                        console.print(f"[dim]   ✓ Completed in {elapsed:.2f}s[/dim]")
                
                elif current_step.status == "failed":
                    console.print(f"[dim]   ✗ Failed: {current_step.error}[/dim]")
            
            # Execute the request with LLM orchestration
            result = await orchestrator.execute_request(
                message,
                context={
                    "session_id": id(self),
                    "conversation_history": self.conversation_history
                },
                progress_callback=progress_callback
            )
            
            # Format and display response
            if result["status"] == "completed":
                workflow_data = result["workflow"]
                execution_summary = result["execution_summary"]
                
                # Show LLM analysis only for multi-step workflows
                if (execution_summary["total_steps"] > 1 and 
                    workflow_data.get("analysis") and 
                    workflow_data["analysis"] != "No analysis provided"):
                    console.print(f"\n[dim]📊 Analysis: {workflow_data['analysis']}[/dim]\n")
                
                # Create a natural response based on the results
                response = await self._format_workflow_response(workflow_data, result["results"])
                
                # Show execution summary if multiple steps
                if execution_summary["total_steps"] > 1:
                    success_rate = execution_summary["success_rate"]
                    exec_time = execution_summary["total_execution_time"]
                    console.print(f"[dim]✅ Completed {execution_summary['completed_steps']}/{execution_summary['total_steps']} steps ({success_rate:.1f}% success) in {exec_time:.1f}s[/dim]")
                
            else:
                # Handle workflow failure
                error_msg = result.get("error", "Unknown error")
                response = f"I apologize, but I encountered an error while processing your request: {error_msg}"
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            console.print(f"{response}")
            
        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")
            console.print(f"[error]Failed to process message: {e}[/error]")
            # Fallback to simple response
            response = "I'm sorry, I'm having trouble processing your request right now. Please try again or use a command like /help for assistance."
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response, 
                "timestamp": datetime.utcnow().isoformat()
            })
            
            console.print(f"{response}")
            
        # Ensure we continue the conversation loop
        # Force garbage collection to clean up any dagger resources
        import gc
        gc.collect()
    
    async def _format_workflow_response(self, workflow_data: dict, step_results: list) -> str:
        """Format workflow results into a natural language response."""
        analysis = workflow_data.get("analysis", "")
        expected_outcome = workflow_data.get("expected_outcome", "")
        
        # Start with analysis
        response_parts = []
        
        # Don't duplicate analysis since we already showed it above
        
        # Extract key results from successful steps
        successful_results = []
        for step_result in step_results:
            if step_result.get("success") and step_result.get("result"):
                result_data = step_result["result"]
                tool_name = step_result["step"]["tool_name"]
                
                # Format results based on tool type
                if tool_name == "thinking":
                    thinking_result = result_data.get("result", {})
                    if isinstance(thinking_result, dict):
                        # Display the recommendations which contain the user's answer
                        if "recommendations" in thinking_result and thinking_result["recommendations"]:
                            recommendations = thinking_result["recommendations"]
                            if isinstance(recommendations, list):
                                # Join list items with newlines for better formatting
                                formatted_recommendations = "\n".join(f"• {rec}" for rec in recommendations)
                                successful_results.append(formatted_recommendations)
                            else:
                                successful_results.append(str(recommendations))
                        
                        # Also include final insight if available and different from recommendations
                        elif "final_insight" in thinking_result and thinking_result["final_insight"]:
                            successful_results.append(thinking_result["final_insight"])
                
                elif tool_name == "file_operations":
                    file_result = result_data.get("result", {})
                    if isinstance(file_result, dict):
                        # Check step parameters to understand what was done
                        step_params = step_result.get("step", {}).get("parameters", {})
                        operation = step_params.get("operation", "")
                        
                        if operation == "write_file" and "content" in step_params:
                            # For write operations, show the content that was written
                            content = step_params["content"]
                            file_path = file_result.get("path", step_params.get("path", ""))
                            successful_results.append(f"Created script at {file_path}:\n```bash\n{content}\n```")
                        elif operation == "read_file" and "content" in file_result:
                            # For read operations, show the content that was read
                            content = file_result["content"]
                            successful_results.append(f"File content:\n```\n{content}\n```")
                        elif "files" in file_result:
                            # For list operations
                            file_count = len(file_result["files"])
                            successful_results.append(f"Found {file_count} files")
                        elif "bytes_written" in file_result:
                            # Generic write confirmation
                            file_path = file_result.get("path", "")
                            successful_results.append(f"File written successfully to: {file_path}")
                
                elif tool_name == "execution":
                    exec_result = result_data.get("result", {})
                    if isinstance(exec_result, dict):
                        # Check for stdout (correct field name from execution.py)
                        if "stdout" in exec_result:
                            stdout = exec_result["stdout"]
                            if stdout and len(stdout.strip()) > 0:
                                successful_results.append(f"Script output:\n```\n{stdout.strip()}\n```")
                        if "stderr" in exec_result:
                            stderr = exec_result["stderr"]
                            if stderr and len(stderr.strip()) > 0:
                                successful_results.append(f"Error output:\n```\n{stderr.strip()}\n```")
                        if "exit_code" in exec_result:
                            exit_code = exec_result["exit_code"]
                            if exit_code == 0:
                                successful_results.append("✓ Script executed successfully")
                            else:
                                successful_results.append(f"⚠ Script exited with code {exit_code}")
        
        # Combine response parts
        if successful_results:
            response_parts.extend(successful_results)
        
        if expected_outcome and expected_outcome != "":
            if not successful_results:  # Only add if we don't have specific results
                response_parts.append(f"Expected outcome: {expected_outcome}")
        
        if not response_parts:
            response_parts.append("I've processed your request, but didn't generate specific output to display.")
        
        return " ".join(response_parts)
    
    def _show_help(self):
        """Show help information."""
        help_text = """[bold yellow]AIDA Interactive Commands[/bold yellow]

[cyan]General Commands:[/cyan]
• [white]/help[/white] or [white]/h[/white] - Show this help
• [white]/status[/white] or [white]/s[/white] - Show system status
• [white]/clear[/white] or [white]/c[/white] - Clear screen
• [white]/exit[/white] or [white]/quit[/white] or [white]/q[/white] - Exit

[cyan]Agent Management:[/cyan] 
• [white]/agents[/white] or [white]/a[/white] - List agents
• [white]/agents create NAME[/white] - Create new agent
• [white]/agents switch NAME[/white] - Switch to agent

[cyan]Tool Management:[/cyan]
• [white]/tools[/white] or [white]/t[/white] - List available tools
• [white]/exec TOOL_NAME --param value[/white] - Execute tool

[cyan]LLM Management:[/cyan]
• [white]/llm[/white] or [white]/l[/white] - List LLM providers
• [white]/llm status[/white] - Check provider health

[cyan]Conversation:[/cyan]
• [white]/history[/white] or [white]/hist[/white] - Show conversation history
• [white]/config[/white] - Show current configuration

[dim]Type any message to chat with AIDA.[/dim]"""
        
        console.print_panel(help_text, title="Help", border_style="yellow")
    
    async def _show_status(self):
        """Show system status."""
        table = Table(title="AIDA System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Agent status
        agent_status = "Running" if self.agent and self.agent._started else "Stopped"
        agent_details = f"ID: {self.agent.agent_id[:8]}..." if self.agent else "None"
        table.add_row("Current Agent", agent_status, agent_details)
        
        # Tool registry
        tools_count = len(await self.tool_registry.list_tools())
        table.add_row("Tool Registry", "Active", f"{tools_count} tools loaded")
        
        # LLM providers
        providers_count = len(self.llm_manager.list_providers())
        table.add_row("LLM Providers", "Ready", f"{providers_count} providers configured")
        
        # Session info
        session_duration = datetime.utcnow() - self.session_started
        table.add_row("Session", "Active", f"Duration: {session_duration}")
        
        # Conversation
        msg_count = len(self.conversation_history)
        table.add_row("Conversation", "Active", f"{msg_count} messages")
        
        console.print_table(table)
    
    async def _manage_agents(self, args: List[str]):
        """Manage agents."""
        if not args:
            # List current agent
            if self.agent:
                console.print(f"[agent]Current agent:[/agent] {self.agent.name}")
                console.print(f"[dim]ID: {self.agent.agent_id}[/dim]")
                console.print(f"[dim]Status: {'Running' if self.agent._started else 'Stopped'}[/dim]")
            else:
                console.print("[warning]No active agent[/warning]")
            return
        
        subcommand = args[0].lower()
        
        if subcommand == "create" and len(args) > 1:
            agent_name = args[1]
            await self._create_agent(agent_name)
        elif subcommand == "switch" and len(args) > 1:
            agent_name = args[1]
            console.print(f"[warning]Agent switching not implemented yet[/warning]")
        else:
            console.print("[warning]Usage: /agents [create|switch] NAME[/warning]")
    
    async def _create_agent(self, name: str):
        """Create a new agent."""
        console.print(f"[yellow]Creating agent: {name}[/yellow]")
        
        description = console.prompt("Agent description", default="Custom interactive agent")
        
        config = AgentConfig(
            name=name,
            description=description,
            capabilities=[
                AgentCapability(
                    name="custom_capability",
                    description="Custom agent capability"
                )
            ]
        )
        
        try:
            new_agent = Agent(config)
            await new_agent.start()
            
            # Stop current agent
            if self.agent:
                await self.agent.stop()
            
            self.agent = new_agent
            console.print(f"[success]✅ Agent '{name}' created and activated[/success]")
            
        except Exception as e:
            console.print(f"[error]Failed to create agent: {e}[/error]")
    
    async def _show_tools(self):
        """Show available tools."""
        tools = await self.tool_registry.list_tools()
        
        if not tools:
            console.print("[warning]No tools available[/warning]")
            return
        
        table = Table(title="Available Tools")
        table.add_column("Tool Name", style="tool")
        table.add_column("Description")
        table.add_column("Status", style="green")
        
        for tool_name in tools:
            tool = await self.tool_registry.get_tool(tool_name)
            if tool:
                table.add_row(
                    tool_name,
                    tool.description,
                    "Ready"
                )
        
        console.print_table(table)
    
    async def _manage_llm(self, args: List[str]):
        """Manage LLM providers."""
        if not args:
            # List providers
            providers = self.llm_manager.list_providers()
            
            if not providers:
                console.print("[warning]No LLM providers configured[/warning]")
                return
            
            table = Table(title="LLM Providers")
            table.add_column("Provider", style="provider")
            table.add_column("Model")
            table.add_column("Type")
            table.add_column("Status")
            
            for provider_key in providers:
                provider = self.llm_manager.get_provider(provider_key)
                if provider:
                    info = provider.get_model_info()
                    table.add_row(
                        provider.provider_name,
                        provider.model,
                        info.get("type", "unknown"),
                        "Ready"
                    )
            
            console.print_table(table)
            return
        
        subcommand = args[0].lower()
        
        if subcommand == "status":
            console.print("[yellow]Checking provider health...[/yellow]")
            health_status = await self.llm_manager.health_check()
            
            for provider_key, is_healthy in health_status.items():
                status = "[green]✅ Healthy[/green]" if is_healthy else "[red]❌ Unhealthy[/red]"
                console.print(f"  {provider_key}: {status}")
        else:
            console.print("[warning]Usage: /llm [status][/warning]")
    
    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            console.print("[dim]No conversation history[/dim]")
            return
        
        console.print(f"[bold]Conversation History ({len(self.conversation_history)} messages)[/bold]\n")
        
        for msg in self.conversation_history[-10:]:  # Show last 10 messages
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"]
            
            if role == "user":
                console.print(f"[cyan]You:[/cyan] {content}")
            else:
                console.print(f"[blue]AIDA:[/blue] {content}")
            
            console.print(f"[dim]{timestamp}[/dim]\n")
    
    async def _execute_tool(self, args: List[str]):
        """Execute a tool."""
        if not args:
            console.print("[warning]Usage: /exec TOOL_NAME --param value[/warning]")
            return
        
        tool_name = args[0]
        tool = await self.tool_registry.get_tool(tool_name)
        
        if not tool:
            console.print(f"[error]Tool '{tool_name}' not found[/error]")
            return
        
        # Parse simple parameters (--param value)
        params = {}
        i = 1
        while i < len(args) - 1:
            if args[i].startswith('--'):
                param_name = args[i][2:]
                param_value = args[i + 1]
                params[param_name] = param_value
                i += 2
            else:
                i += 1
        
        try:
            console.print(f"[yellow]Executing tool: {tool_name}[/yellow]")
            result = await tool.execute_async(**params)
            
            console.print(f"[success]✅ Tool executed successfully[/success]")
            console.print_json(result.result)
            
        except Exception as e:
            console.print(f"[error]Tool execution failed: {e}[/error]")
    
    async def _show_config(self):
        """Show current configuration."""
        config_info = {
            "session": {
                "started": self.session_started.isoformat(),
                "duration": str(datetime.utcnow() - self.session_started),
                "messages": len(self.conversation_history)
            },
            "agent": {
                "name": self.agent.name if self.agent else None,
                "id": self.agent.agent_id if self.agent else None,
                "status": "running" if self.agent and self.agent._started else "stopped"
            },
            "tools": {
                "count": len(await self.tool_registry.list_tools())
            },
            "llm": {
                "providers": len(self.llm_manager.list_providers())
            }
        }
        
        console.print_json(config_info)
    
    async def _cleanup(self):
        """Cleanup session resources."""
        if self.agent:
            await self.agent.stop()
        
        console.print("[dim]Session cleanup completed[/dim]")


@interactive_app.command()
def start():
    """Start AIDA interactive mode."""
    async def _start():
        session = InteractiveSession()
        await session.start()
    
    try:
        asyncio.run(_start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except Exception as e:
        console.print(f"[error]Interactive mode failed: {e}[/error]")


# Default command when called without subcommand
@interactive_app.callback(invoke_without_command=True)
def interactive_main(ctx: typer.Context):
    """Start AIDA interactive mode."""
    if ctx.invoked_subcommand is None:
        start()