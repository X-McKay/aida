"""Storage management for coordinator plans.

Provides persistent storage for TodoPlans in the coordinator-worker architecture,
with support for plan lifecycle management and archival.
"""

import contextlib
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import shutil
from typing import Any

from aida.agents.coordination.plan_models import TodoPlan

logger = logging.getLogger(__name__)


class CoordinatorPlanStorage:
    """Manages persistent storage and retrieval of plans for the coordinator.

    Handles plan lifecycle including creation, archival, and cleanup operations
    with support for concurrent access and atomic operations.
    """

    def __init__(self, storage_dir: str = ".aida/orchestrator"):
        """Initialize plan storage with specified directory.

        Args:
            storage_dir: Directory to store plan files
        """
        self.storage_dir = Path(storage_dir)
        self._ensure_storage_dirs()

    def _ensure_storage_dirs(self) -> None:
        """Ensure storage directories exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "active").mkdir(exist_ok=True)
        (self.storage_dir / "archived").mkdir(exist_ok=True)
        (self.storage_dir / "failed").mkdir(exist_ok=True)

    def save_plan(self, plan: TodoPlan, subdir: str = "active") -> Path:
        """Save a plan to storage.

        Args:
            plan: The TodoPlan to save
            subdir: Subdirectory (active, archived, failed)

        Returns:
            Path to the saved file
        """
        filename = f"plan_{plan.id}.json"
        filepath = self.storage_dir / subdir / filename

        try:
            # Ensure plan has all required fields for serialization
            plan_data = {
                "id": plan.id,
                "user_request": plan.user_request,
                "analysis": plan.analysis,
                "expected_outcome": plan.expected_outcome,
                "context": plan.context,
                "created_at": plan.created_at.isoformat(),
                "last_updated": plan.last_updated.isoformat(),
                "plan_version": plan.plan_version,
                "steps": [
                    {
                        "id": step.id,
                        "description": step.description,
                        "tool_name": step.tool_name,
                        "parameters": step.parameters,
                        "dependencies": step.dependencies,
                        "status": step.status.value,
                        "result": step.result,
                        "error": step.error,
                        "started_at": step.started_at.isoformat() if step.started_at else None,
                        "completed_at": step.completed_at.isoformat()
                        if step.completed_at
                        else None,
                        "retry_count": step.retry_count,
                        "max_retries": step.max_retries,
                    }
                    for step in plan.steps
                ],
            }

            with open(filepath, "w") as f:
                json.dump(plan_data, f, indent=2)

            logger.info(f"Saved plan {plan.id} to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save plan {plan.id}: {e}")
            raise

    def load_plan(self, plan_id: str) -> TodoPlan | None:
        """Load a plan by ID from any subdirectory.

        Args:
            plan_id: The plan ID to load

        Returns:
            TodoPlan if found, None otherwise
        """
        filename = f"plan_{plan_id}.json"

        # Check all subdirectories
        for subdir in ["active", "archived", "failed"]:
            filepath = self.storage_dir / subdir / filename
            if filepath.exists():
                return self._load_plan_from_file(filepath)

        return None

    def _load_plan_from_file(self, filepath: Path) -> TodoPlan | None:
        """Load a plan from a specific file."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Convert back to TodoPlan using Pydantic
            try:
                return TodoPlan(**data)  # type: ignore[missing-argument]
            except Exception as e:
                logger.error(f"Failed to parse plan data: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to load plan from {filepath}: {e}")
            return None

    def list_plans(self, subdir: str | None = None) -> list[dict[str, Any]]:
        """List all plans with metadata.

        Args:
            subdir: Specific subdirectory to list, or None for all

        Returns:
            List of plan metadata dictionaries
        """
        plans = []

        # Determine which directories to search
        if subdir:
            search_dirs = [self.storage_dir / subdir]
        else:
            search_dirs = [
                self.storage_dir / "active",
                self.storage_dir / "archived",
                self.storage_dir / "failed",
            ]

        for directory in search_dirs:
            if not directory.exists():
                continue

            for filepath in directory.glob("plan_*.json"):
                try:
                    with open(filepath) as f:
                        data = json.load(f)

                    # Calculate progress
                    total_steps = len(data.get("steps", []))
                    completed_steps = sum(
                        1 for step in data.get("steps", []) if step.get("status") == "completed"
                    )
                    failed_steps = sum(
                        1 for step in data.get("steps", []) if step.get("status") == "failed"
                    )

                    # Determine overall status
                    if failed_steps > 0:
                        status = "failed"
                    elif completed_steps == total_steps:
                        status = "completed"
                    elif completed_steps > 0:
                        status = "in_progress"
                    else:
                        status = "pending"

                    plans.append(
                        {
                            "id": data["id"],
                            "user_request": data["user_request"],
                            "created_at": data["created_at"],
                            "last_updated": data.get("last_updated", data["created_at"]),
                            "status": status,
                            "steps_total": total_steps,
                            "steps_completed": completed_steps,
                            "steps_failed": failed_steps,
                            "directory": directory.name,
                            "filepath": str(filepath),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to read plan metadata from {filepath}: {e}")

        # Sort by last updated, newest first
        plans.sort(key=lambda x: x["last_updated"], reverse=True)
        return plans

    def move_plan(self, plan_id: str, to_subdir: str) -> bool:
        """Move a plan between subdirectories.

        Args:
            plan_id: Plan ID to move
            to_subdir: Target subdirectory (active, archived, failed)

        Returns:
            True if successful, False otherwise
        """
        filename = f"plan_{plan_id}.json"

        # Find current location
        source_path = None
        for subdir in ["active", "archived", "failed"]:
            path = self.storage_dir / subdir / filename
            if path.exists():
                source_path = path
                break

        if not source_path:
            logger.warning(f"Plan {plan_id} not found")
            return False

        # Move to new location
        target_path = self.storage_dir / to_subdir / filename

        try:
            shutil.move(str(source_path), str(target_path))
            logger.info(f"Moved plan {plan_id} to {to_subdir}")
            return True
        except Exception as e:
            logger.error(f"Failed to move plan {plan_id}: {e}")
            return False

    def archive_completed_plans(self) -> int:
        """Archive all completed plans from active directory.

        Returns:
            Number of plans archived
        """
        archived_count = 0

        for plan_info in self.list_plans("active"):
            if plan_info["status"] == "completed" and self.move_plan(plan_info["id"], "archived"):
                archived_count += 1

        logger.info(f"Archived {archived_count} completed plans")
        return archived_count

    def cleanup_old_plans(self, days_old: int = 30, delete_archived: bool = True) -> int:
        """Remove plans older than specified days.

        Args:
            days_old: Age threshold in days
            delete_archived: Whether to also delete from archived

        Returns:
            Number of plans deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        deleted_count = 0

        # Determine which directories to clean
        subdirs = ["failed"]
        if delete_archived:
            subdirs.append("archived")

        for subdir in subdirs:
            for plan_info in self.list_plans(subdir):
                plan_date = datetime.fromisoformat(plan_info["created_at"].replace("Z", "+00:00"))

                if plan_date < cutoff_date:
                    try:
                        Path(plan_info["filepath"]).unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old plan {plan_info['id']}")
                    except Exception as e:
                        logger.error(f"Failed to delete plan {plan_info['id']}: {e}")

        return deleted_count

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "storage_dir": str(self.storage_dir),
            "total_plans": 0,
            "total_size_bytes": 0,
            "by_status": {},
            "by_directory": {},
        }

        # Count plans and calculate sizes
        for subdir in ["active", "archived", "failed"]:
            dir_path = self.storage_dir / subdir
            if not dir_path.exists():
                continue

            dir_count = 0
            dir_size = 0

            for filepath in dir_path.glob("plan_*.json"):
                dir_count += 1
                with contextlib.suppress(OSError):
                    dir_size += filepath.stat().st_size

            stats["by_directory"][subdir] = {"count": dir_count, "size_bytes": dir_size}
            stats["total_plans"] += dir_count
            stats["total_size_bytes"] += dir_size

        # Count by status
        all_plans = self.list_plans()
        for plan in all_plans:
            status = plan["status"]
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        # Find oldest and newest
        if all_plans:
            stats["oldest_plan"] = min(p["created_at"] for p in all_plans)
            stats["newest_plan"] = max(p["created_at"] for p in all_plans)
        else:
            stats["oldest_plan"] = None
            stats["newest_plan"] = None

        return stats

    def export_summary_report(self, output_file: str | None = None) -> str:
        """Export a summary report of all plans.

        Args:
            output_file: Output filename, auto-generated if None

        Returns:
            Path to the exported file
        """
        if output_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"coordinator_plan_summary_{timestamp}.txt"

        plans = self.list_plans()
        stats = self.get_storage_stats()

        lines = [
            "Coordinator Plan Summary Report",
            "=" * 60,
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Storage Directory: {stats['storage_dir']}",
            f"Total Plans: {stats['total_plans']}",
            f"Total Size: {stats['total_size_bytes'] / 1024:.1f} KB",
            "",
            "Status Breakdown:",
            "-" * 30,
        ]

        # Status summary
        for status, count in stats["by_status"].items():
            lines.append(f"  {status.capitalize()}: {count}")

        lines.extend(["", "Directory Breakdown:", "-" * 30])

        # Directory summary
        for dir_name, dir_stats in stats["by_directory"].items():
            lines.append(
                f"  {dir_name}: {dir_stats['count']} plans ({dir_stats['size_bytes'] / 1024:.1f} KB)"
            )

        lines.extend(["", "Recent Plans:", "-" * 30])

        # List recent plans (up to 20)
        for plan in plans[:20]:
            lines.append(f"\n{plan['id']}:")
            lines.append(f"  Request: {plan['user_request'][:80]}...")
            lines.append(f"  Created: {plan['created_at']}")
            lines.append(f"  Status: {plan['status']}")
            lines.append(f"  Progress: {plan['steps_completed']}/{plan['steps_total']} steps")
            if plan["steps_failed"] > 0:
                lines.append(f"  Failed: {plan['steps_failed']} steps")

        content = "\n".join(lines)

        try:
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Exported plan summary to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to export summary: {e}")
            raise
