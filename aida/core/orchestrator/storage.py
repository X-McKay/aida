"""Storage management for TODO plans."""

import contextlib
from datetime import datetime, timedelta
import glob
import logging
import os
import shutil
from typing import Any

from .models import TodoPlan

logger = logging.getLogger(__name__)


class PlanStorageManager:
    """Manages persistent storage and retrieval of TodoPlans."""

    def __init__(self, storage_dir: str = ".aida/plans"):
        self.storage_dir = storage_dir
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """Ensure the storage directory exists."""
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_plan(self, plan: TodoPlan) -> str:
        """Save a plan to storage and return the file path."""
        filename = f"plan_{plan.id}.json"
        filepath = os.path.join(self.storage_dir, filename)

        try:
            with open(filepath, "w") as f:
                f.write(plan.to_json())
            logger.info(f"Saved plan {plan.id} to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save plan {plan.id}: {e}")
            raise

    def load_plan(self, plan_id: str) -> TodoPlan | None:
        """Load a plan from storage by ID."""
        filename = f"plan_{plan_id}.json"
        filepath = os.path.join(self.storage_dir, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath) as f:
                content = f.read()
            return TodoPlan.from_json(content)
        except Exception as e:
            logger.error(f"Failed to load plan {plan_id}: {e}")
            return None

    def list_plans(self) -> list[dict[str, Any]]:
        """List all stored plans with basic metadata."""
        plans = []
        pattern = os.path.join(self.storage_dir, "plan_*.json")

        for filepath in glob.glob(pattern):
            try:
                with open(filepath) as f:
                    content = f.read()
                plan = TodoPlan.from_json(content)

                plans.append(
                    {
                        "id": plan.id,
                        "user_request": plan.user_request,
                        "created_at": plan.created_at.isoformat(),
                        "status": plan.get_progress()["status"],
                        "steps_total": len(plan.steps),
                        "steps_completed": plan.get_progress()["completed"],
                        "filepath": filepath,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to read plan metadata from {filepath}: {e}")

        # Sort by creation time, newest first
        plans.sort(key=lambda x: x["created_at"], reverse=True)
        return plans

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan from storage."""
        filename = f"plan_{plan_id}.json"
        filepath = os.path.join(self.storage_dir, filename)

        if not os.path.exists(filepath):
            return False

        try:
            os.remove(filepath)
            logger.info(f"Deleted plan {plan_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete plan {plan_id}: {e}")
            return False

    def archive_completed_plans(self, archive_dir: str = None) -> int:
        """Archive completed plans to a separate directory."""
        if archive_dir is None:
            archive_dir = os.path.join(self.storage_dir, "archived")

        os.makedirs(archive_dir, exist_ok=True)
        archived_count = 0

        for plan_info in self.list_plans():
            if plan_info["status"] == "completed":
                source = plan_info["filepath"]
                filename = os.path.basename(source)
                destination = os.path.join(archive_dir, filename)

                try:
                    shutil.move(source, destination)
                    archived_count += 1
                    logger.info(f"Archived completed plan {plan_info['id']}")
                except Exception as e:
                    logger.error(f"Failed to archive plan {plan_info['id']}: {e}")

        return archived_count

    def cleanup_old_plans(self, days_old: int = 30) -> int:
        """Clean up plans older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        deleted_count = 0

        for plan_info in self.list_plans():
            plan_date = datetime.fromisoformat(plan_info["created_at"].replace("Z", "+00:00"))

            if plan_date < cutoff_date and self.delete_plan(plan_info["id"]):
                deleted_count += 1

        return deleted_count

    def export_plan_summary(self, output_file: str = None) -> str:
        """Export a summary of all plans to a text file."""
        if output_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"plan_summary_{timestamp}.txt"

        plans = self.list_plans()

        lines = [
            "TODO Plan Summary Report",
            "=" * 50,
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Total Plans: {len(plans)}",
            "",
        ]

        # Group by status
        status_groups = {}
        for plan in plans:
            status = plan["status"]
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(plan)

        for status, status_plans in status_groups.items():
            lines.append(f"{status.upper()} PLANS ({len(status_plans)}):")
            lines.append("-" * 30)

            for plan in status_plans:
                lines.append(f"  {plan['id']}: {plan['user_request'][:60]}...")
                lines.append(f"    Created: {plan['created_at']}")
                lines.append(f"    Progress: {plan['steps_completed']}/{plan['steps_total']}")
                lines.append("")

        content = "\n".join(lines)

        try:
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Exported plan summary to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to export plan summary: {e}")
            raise

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        plans = self.list_plans()

        # Calculate size
        total_size = 0
        for plan_info in plans:
            with contextlib.suppress(OSError):
                total_size += os.path.getsize(plan_info["filepath"])

        # Group by status
        status_counts = {}
        for plan in plans:
            status = plan["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_plans": len(plans),
            "total_size_bytes": total_size,
            "storage_dir": self.storage_dir,
            "status_breakdown": status_counts,
            "oldest_plan": plans[-1]["created_at"] if plans else None,
            "newest_plan": plans[0]["created_at"] if plans else None,
        }
