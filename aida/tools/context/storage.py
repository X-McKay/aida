"""Storage management for context snapshots."""

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any

from .models import ContextSnapshot

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages context snapshots."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def create_snapshot(self, content: Any, file_path: str | None = None) -> ContextSnapshot:
        """Create a new snapshot."""
        snapshot = ContextSnapshot(
            content=content if isinstance(content, dict) else {"content": content},
            metadata={"created_by": "context_tool", "content_type": type(content).__name__},
        )

        # Determine file path
        save_path = Path(file_path) if file_path else self.storage_dir / f"{snapshot.id}.json"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save snapshot
        snapshot_data = snapshot.dict()
        save_path.write_text(json.dumps(snapshot_data, indent=2, default=str))

        snapshot.file_path = str(save_path.absolute())

        logger.info(f"Created snapshot {snapshot.id} at {snapshot.file_path}")

        return snapshot

    async def load_snapshot(self, snapshot_id: str) -> ContextSnapshot | None:
        """Load a snapshot by ID or path."""
        # Check if it's a full path
        if snapshot_id.endswith(".json"):
            snapshot_path = Path(snapshot_id)
            if not snapshot_path.exists():
                logger.warning(f"Snapshot {snapshot_id} not found")
                return None
        else:
            # Try to find snapshot file
            snapshot_files = list(self.storage_dir.glob(f"{snapshot_id}*.json"))

            if not snapshot_files:
                logger.warning(f"Snapshot {snapshot_id} not found")
                return None

            snapshot_path = snapshot_files[0]

        try:
            snapshot_data = json.loads(snapshot_path.read_text())
            return ContextSnapshot(**snapshot_data)
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None

    async def list_snapshots(self, limit: int | None = None) -> list[ContextSnapshot]:
        """List available snapshots."""
        snapshots = []

        for snapshot_file in self.storage_dir.glob("*.json"):
            try:
                snapshot_data = json.loads(snapshot_file.read_text())
                snapshots.append(ContextSnapshot(**snapshot_data))
            except Exception as e:
                logger.warning(f"Failed to load snapshot from {snapshot_file}: {e}")

        # Sort by creation date (newest first)
        snapshots.sort(key=lambda s: s.created_at, reverse=True)

        if limit:
            snapshots = snapshots[:limit]

        return snapshots

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        snapshot_files = list(self.storage_dir.glob(f"{snapshot_id}*.json"))

        if not snapshot_files:
            return False

        for file in snapshot_files:
            file.unlink()
            logger.info(f"Deleted snapshot file: {file}")

        return True

    async def cleanup_old_snapshots(self, retention_days: int) -> int:
        """Clean up snapshots older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        deleted_count = 0

        for snapshot_file in self.storage_dir.glob("*.json"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(snapshot_file.stat().st_mtime)

                if mtime < cutoff_date:
                    snapshot_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old snapshot: {snapshot_file}")

            except Exception as e:
                logger.warning(f"Failed to clean up snapshot {snapshot_file}: {e}")

        return deleted_count

    def get_storage_info(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for snapshot_file in self.storage_dir.glob("*.json"):
            total_size += snapshot_file.stat().st_size
            file_count += 1

        return {
            "storage_dir": str(self.storage_dir),
            "snapshot_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
