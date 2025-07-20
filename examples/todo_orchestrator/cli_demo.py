"""Demo command showing the new TODO-based orchestrator."""

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from aida.core.orchestrator import get_todo_orchestrator, TodoPlan, TodoStep, ReplanReason


console = Console()
todo_app = typer.Typer(name="todo", help="TODO-based workflow orchestration commands")


@todo_app.command("demo")
def todo_demo(
    request: str = typer.Argument(..., help="Request to process"),
    auto_replan: bool = typer.Option(True, "--auto-replan/--no-auto-replan", help="Automatically replan on failures"),
    show_live: bool = typer.Option(True, "--show-live/--no-show-live", help="Show live progress updates")
):
    """
    Demo the new TODO-based orchestrator.
    
    Examples:
        aida todo demo "Create a Python script that calculates fibonacci numbers"
        aida todo demo "Analyze the performance of sorting algorithms"
    """
    
    async def _todo_demo():
        console.print(f"🚀 Starting TODO-based workflow for: [bold]{request}[/bold]")
        
        # Get orchestrator
        orchestrator = get_todo_orchestrator()
        
        try:
            # Create plan
            console.print("\n📋 Creating workflow plan...")
            plan = await orchestrator.create_plan(request)
        
        # Show initial plan
        console.print("\n" + "="*60)
        console.print(Markdown(plan.to_markdown()))
        console.print("="*60)
        
        # Progress tracking
        def progress_callback(plan: TodoPlan, step: TodoStep):
            """Called when a step starts executing."""
            console.print(f"\n🔄 Executing: [yellow]{step.description}[/yellow]")
        
        def replan_callback(plan: TodoPlan, reason: ReplanReason) -> bool:
            """Called when replanning is needed."""
            if not auto_replan:
                return typer.confirm(f"\nReplanning needed ({reason.value}). Continue?")
            
            console.print(f"\n🔄 Auto-replanning due to: [red]{reason.value}[/red]")
            return True
        
        # Execute plan
        if show_live:
            # Live updating display
            with Live(console=console, refresh_per_second=2) as live:
                async def live_execute():
                    result = await orchestrator.execute_plan(
                        plan, 
                        progress_callback=lambda p, s: None,  # Don't print during live
                        replan_callback=replan_callback
                    )
                    return result
                
                # Update display periodically
                async def update_display():
                    while True:
                        try:
                            markdown_content = plan.to_markdown()
                            progress = plan.get_progress()
                            
                            panel_content = [
                                Markdown(markdown_content),
                                f"\n[bold]Status:[/bold] {progress['status']}",
                                f"[bold]Progress:[/bold] {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)"
                            ]
                            
                            if plan.plan_version > 1:
                                panel_content.append(f"[bold]Plan Version:[/bold] {plan.plan_version}")
                            
                            live.update(Panel(
                                "\n".join(panel_content),
                                title="TODO Workflow Progress",
                                border_style="blue"
                            ))
                            
                            await asyncio.sleep(0.5)
                        except asyncio.CancelledError:
                            break
                
                # Run both tasks
                display_task = asyncio.create_task(update_display())
                execution_task = asyncio.create_task(live_execute())
                
                try:
                    result = await execution_task
                finally:
                    display_task.cancel()
                    try:
                        await display_task
                    except asyncio.CancelledError:
                        pass
        else:
            # Simple execution with callbacks
            result = await orchestrator.execute_plan(
                plan,
                progress_callback=progress_callback,
                replan_callback=replan_callback
            )
        
        # Show final results
        console.print("\n" + "="*60)
        console.print("[bold green]FINAL RESULTS[/bold green]")
        console.print("="*60)
        
        console.print(Markdown(result["final_markdown"]))
        
        # Show execution summary
        progress = plan.get_progress()
        console.print(f"\n[bold]Final Status:[/bold] {result['status']}")
        console.print(f"[bold]Steps Completed:[/bold] {progress['completed']}/{progress['total']}")
        console.print(f"[bold]Plan Versions:[/bold] {plan.plan_version}")
        
        if plan.replan_history:
            console.print(f"[bold]Replanning Events:[/bold] {len(plan.replan_history)}")
            for i, replan in enumerate(plan.replan_history, 1):
                console.print(f"  {i}. {replan['reason']} at version {replan['old_version']}")
        
        # Show any failed steps
        failed_steps = [s for s in plan.steps if s.status.value == "failed"]
        if failed_steps:
            console.print(f"\n[bold red]Failed Steps:[/bold red]")
            for step in failed_steps:
                console.print(f"  ❌ {step.description}: {step.error}")
        
            console.print(f"\n✅ Workflow {'completed successfully' if result['status'] == 'completed' else 'finished with issues'}!")
            
        except Exception as e:
            console.print(f"\n❌ Error: {e}")
            raise
    
    asyncio.run(_todo_demo())


@todo_app.command("list")
def list_plans(
    format: str = typer.Option("table", help="Output format for plan list", case_sensitive=False)
):
    """List all active TODO plans."""
    # Validate format
    valid_formats = ["table", "markdown", "json"]
    if format.lower() not in valid_formats:
        console.print(f"[red]Invalid format '{format}'. Valid options: {', '.join(valid_formats)}[/red]")
        raise typer.Exit(1)
    
    format = format.lower()
    
    orchestrator = get_todo_orchestrator()
    plans = orchestrator.list_plans()
    
    if not plans:
        console.print("No active plans found.")
        return
    
    if format == 'json':
        console.print(json.dumps(plans, indent=2))
    elif format == 'markdown':
        for plan in plans:
            markdown = orchestrator.get_plan_markdown(plan['id'])
            if markdown:
                console.print(Markdown(markdown))
                console.print("\n" + "-"*40 + "\n")
    else:
        # Table format
        
        table = Table(title="Active TODO Plans")
        table.add_column("ID", style="cyan")
        table.add_column("Request", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Version", style="blue")
        table.add_column("Created", style="dim")
        
        for plan in plans:
            table.add_row(
                plan['id'][:8] + "...",
                plan['user_request'][:40] + "..." if len(plan['user_request']) > 40 else plan['user_request'],
                plan['status'],
                f"{plan['progress']:.1f}%",
                str(plan['version']),
                plan['created_at'][:19]  # Remove microseconds
            )
        
        console.print(table)


@todo_app.command("show")
def show_plan(
    plan_id: str = typer.Argument(..., help="Plan ID to show")
):
    """Show detailed view of a specific plan."""
    orchestrator = get_todo_orchestrator()
    markdown = orchestrator.get_plan_markdown(plan_id)
    
    if not markdown:
        console.print(f"Plan '{plan_id}' not found.")
        raise typer.Exit(1)
    
    console.print(Markdown(markdown))


if __name__ == "__main__":
    todo_app()