"""Supabase checkpointer for LangGraph."""
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)


class SupabaseCheckpointer:
    """Checkpointer implementation using Supabase for LangGraph state persistence."""

    def __init__(self):
        """Initialize Supabase checkpointer."""
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_key,
        )
        self.ttl_seconds = settings.checkpoint_ttl_seconds

    async def put(
        self,
        thread_id: str,
        checkpoint_id: str,
        checkpoint: Dict[str, Any],
        parent_checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a checkpoint to Supabase.

        Args:
            thread_id: Thread/conversation ID
            checkpoint_id: Unique checkpoint ID
            checkpoint: Checkpoint data (state)
            parent_checkpoint_id: Parent checkpoint ID (if any)
            metadata: Additional metadata
        """
        try:
            data = {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint": checkpoint,
                "metadata": metadata or {},
            }
            if parent_checkpoint_id:
                data["parent_checkpoint_id"] = parent_checkpoint_id

            # Upsert checkpoint
            self.client.table("agent_checkpoints").upsert(data).execute()
            logger.debug(f"Saved checkpoint {checkpoint_id} for thread {thread_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    async def get(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a checkpoint from Supabase.

        Args:
            thread_id: Thread/conversation ID
            checkpoint_id: Checkpoint ID (if None, get latest)

        Returns:
            Checkpoint data or None if not found
        """
        try:
            query = self.client.table("agent_checkpoints").select("*").eq("thread_id", thread_id)

            if checkpoint_id:
                query = query.eq("checkpoint_id", checkpoint_id)
            else:
                # Get latest checkpoint
                query = query.order("created_at", desc=True).limit(1)

            result = query.execute()
            if result.data:
                checkpoint_data = result.data[0]
                return {
                    "checkpoint_id": checkpoint_data["checkpoint_id"],
                    "parent_checkpoint_id": checkpoint_data.get("parent_checkpoint_id"),
                    "checkpoint": checkpoint_data["checkpoint"],
                    "metadata": checkpoint_data.get("metadata", {}),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None

    async def list(
        self,
        thread_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List checkpoints for a thread.

        Args:
            thread_id: Thread/conversation ID
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoint data
        """
        try:
            result = (
                self.client.table("agent_checkpoints")
                .select("*")
                .eq("thread_id", thread_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return [
                {
                    "checkpoint_id": cp["checkpoint_id"],
                    "parent_checkpoint_id": cp.get("parent_checkpoint_id"),
                    "checkpoint": cp["checkpoint"],
                    "metadata": cp.get("metadata", {}),
                    "created_at": cp.get("created_at"),
                }
                for cp in result.data
            ]
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> None:
        """
        Delete checkpoint(s) from Supabase.

        Args:
            thread_id: Thread/conversation ID
            checkpoint_id: Specific checkpoint ID (if None, delete all for thread)
        """
        try:
            query = self.client.table("agent_checkpoints").delete().eq("thread_id", thread_id)
            if checkpoint_id:
                query = query.eq("checkpoint_id", checkpoint_id)
            query.execute()
            logger.debug(f"Deleted checkpoint(s) for thread {thread_id}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            raise

    async def cleanup_old_checkpoints(self) -> int:
        """
        Clean up checkpoints older than TTL.

        Returns:
            Number of checkpoints deleted
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.ttl_seconds)
            result = (
                self.client.table("agent_checkpoints")
                .delete()
                .lt("created_at", cutoff_time.isoformat())
                .execute()
            )
            deleted_count = len(result.data) if result.data else 0
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0

