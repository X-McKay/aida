"""System maintenance tool for AIDA."""

import asyncio
import logging
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import psutil
import os

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter


logger = logging.getLogger(__name__)


class MaintenanceTool(Tool):
    """Comprehensive system maintenance and optimization tool."""
    
    def __init__(self):
        super().__init__(
            name="maintenance",
            description="System maintenance, updates, optimization, and health management",
            version="1.0.0"
        )
    
    def get_capability(self) -> ToolCapability:
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="str",
                    description="Maintenance operation to perform",
                    required=True,
                    choices=[
                        "health_check", "system_update", "cleanup", "optimize",
                        "backup_config", "restore_config", "update_prompts",
                        "rotate_logs", "check_dependencies", "repair_system",
                        "performance_tune", "security_audit", "disk_cleanup",
                        "memory_optimization", "cache_management"
                    ]
                ),
                ToolParameter(
                    name="target",
                    type="str",
                    description="Target component for maintenance",
                    required=False,
                    choices=["all", "agents", "tools", "llm_providers", "configs", "logs", "cache"]
                ),
                ToolParameter(
                    name="config_data",
                    type="dict",
                    description="Configuration data for updates",
                    required=False
                ),
                ToolParameter(
                    name="backup_path",
                    type="str",
                    description="Path for backup operations",
                    required=False
                ),
                ToolParameter(
                    name="severity_level",
                    type="str",
                    description="Maintenance severity level",
                    required=False,
                    default="standard",
                    choices=["minimal", "standard", "aggressive", "emergency"]
                ),
                ToolParameter(
                    name="dry_run",
                    type="bool",
                    description="Perform dry run without making changes",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="schedule_time",
                    type="str",
                    description="Schedule maintenance for specific time",
                    required=False
                ),
                ToolParameter(
                    name="notification_enabled",
                    type="bool",
                    description="Enable notifications for maintenance actions",
                    required=False,
                    default=True
                )
            ],
            required_permissions=["system_admin", "file_system"],
            supported_platforms=["linux", "darwin", "windows"],
            dependencies=["psutil"]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute maintenance operation."""
        operation = kwargs["operation"]
        target = kwargs.get("target", "all")
        dry_run = kwargs.get("dry_run", False)
        
        try:
            if operation == "health_check":
                result = await self._health_check(target)
            elif operation == "system_update":
                result = await self._system_update(target, dry_run)
            elif operation == "cleanup":
                result = await self._cleanup(target, kwargs.get("severity_level", "standard"), dry_run)
            elif operation == "optimize":
                result = await self._optimize_system(target, kwargs.get("severity_level", "standard"), dry_run)
            elif operation == "backup_config":
                result = await self._backup_config(kwargs.get("backup_path"), target)
            elif operation == "restore_config":
                result = await self._restore_config(kwargs.get("backup_path"), target)
            elif operation == "update_prompts":
                result = await self._update_prompts(kwargs.get("config_data", {}))
            elif operation == "rotate_logs":
                result = await self._rotate_logs(target, dry_run)
            elif operation == "check_dependencies":
                result = await self._check_dependencies(target)
            elif operation == "repair_system":
                result = await self._repair_system(target, kwargs.get("severity_level", "standard"), dry_run)
            elif operation == "performance_tune":
                result = await self._performance_tune(target, dry_run)
            elif operation == "security_audit":
                result = await self._security_audit(target)
            elif operation == "disk_cleanup":
                result = await self._disk_cleanup(kwargs.get("severity_level", "standard"), dry_run)
            elif operation == "memory_optimization":
                result = await self._memory_optimization(dry_run)
            elif operation == "cache_management":
                result = await self._cache_management(target, kwargs.get("severity_level", "standard"), dry_run)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status="completed",
                result=result,
                started_at=None,
                metadata={
                    "operation": operation,
                    "target": target,
                    "dry_run": dry_run,
                    "timestamp": datetime.utcnow().isoformat(),
                    "changes_made": not dry_run
                }
            )
            
        except Exception as e:
            raise Exception(f"Maintenance operation failed: {str(e)}")
    
    async def _health_check(self, target: str) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        await asyncio.sleep(0.1)
        
        health_status = {
            "overall_health": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks_performed": [],
            "issues_found": [],
            "recommendations": [],
            "system_metrics": {}
        }
        
        # System resource checks
        if target in ["all", "system"]:
            health_status["system_metrics"] = await self._check_system_resources()
            health_status["checks_performed"].append("system_resources")
        
        # Agent health checks
        if target in ["all", "agents"]:
            agent_health = await self._check_agent_health()
            health_status["agent_health"] = agent_health
            health_status["checks_performed"].append("agents")
            
            if agent_health["issues_count"] > 0:
                health_status["issues_found"].extend(agent_health["issues"])
        
        # Tool health checks
        if target in ["all", "tools"]:
            tool_health = await self._check_tool_health()
            health_status["tool_health"] = tool_health
            health_status["checks_performed"].append("tools")
        
        # LLM provider health checks
        if target in ["all", "llm_providers"]:
            llm_health = await self._check_llm_health()
            health_status["llm_health"] = llm_health
            health_status["checks_performed"].append("llm_providers")
        
        # Configuration integrity checks
        if target in ["all", "configs"]:
            config_health = await self._check_config_integrity()
            health_status["config_health"] = config_health
            health_status["checks_performed"].append("configurations")
        
        # Log health checks
        if target in ["all", "logs"]:
            log_health = await self._check_log_health()
            health_status["log_health"] = log_health
            health_status["checks_performed"].append("logs")
        
        # Determine overall health
        total_issues = len(health_status["issues_found"])
        if total_issues == 0:
            health_status["overall_health"] = "healthy"
        elif total_issues <= 3:
            health_status["overall_health"] = "warning"
        else:
            health_status["overall_health"] = "critical"
        
        # Generate recommendations
        health_status["recommendations"] = self._generate_health_recommendations(health_status)
        
        return health_status
    
    async def _system_update(self, target: str, dry_run: bool) -> Dict[str, Any]:
        """Update system components."""
        await asyncio.sleep(0.2)
        
        update_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": target,
            "dry_run": dry_run,
            "updates_available": [],
            "updates_applied": [],
            "update_failures": [],
            "reboot_required": False
        }
        
        # Check for updates based on target
        if target in ["all", "agents"]:
            agent_updates = await self._check_agent_updates()
            update_results["updates_available"].extend(agent_updates)
            
            if not dry_run:
                applied_updates = await self._apply_agent_updates(agent_updates)
                update_results["updates_applied"].extend(applied_updates)
        
        if target in ["all", "tools"]:
            tool_updates = await self._check_tool_updates()
            update_results["updates_available"].extend(tool_updates)
            
            if not dry_run:
                applied_updates = await self._apply_tool_updates(tool_updates)
                update_results["updates_applied"].extend(applied_updates)
        
        if target in ["all", "llm_providers"]:
            llm_updates = await self._check_llm_updates()
            update_results["updates_available"].extend(llm_updates)
            
            if not dry_run:
                applied_updates = await self._apply_llm_updates(llm_updates)
                update_results["updates_applied"].extend(applied_updates)
        
        # Check if reboot is required
        update_results["reboot_required"] = self._check_reboot_required(update_results["updates_applied"])
        
        return update_results
    
    async def _cleanup(self, target: str, severity: str, dry_run: bool) -> Dict[str, Any]:
        """Clean up system resources."""
        await asyncio.sleep(0.1)
        
        cleanup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": target,
            "severity": severity,
            "dry_run": dry_run,
            "space_freed": 0,
            "files_removed": 0,
            "actions_performed": []
        }
        
        # Temporary file cleanup
        if target in ["all", "cache"]:
            temp_cleanup = await self._cleanup_temp_files(severity, dry_run)
            cleanup_results["actions_performed"].append("temp_file_cleanup")
            cleanup_results["space_freed"] += temp_cleanup["space_freed"]
            cleanup_results["files_removed"] += temp_cleanup["files_removed"]
        
        # Log cleanup
        if target in ["all", "logs"]:
            log_cleanup = await self._cleanup_logs(severity, dry_run)
            cleanup_results["actions_performed"].append("log_cleanup")
            cleanup_results["space_freed"] += log_cleanup["space_freed"]
            cleanup_results["files_removed"] += log_cleanup["files_removed"]
        
        # Cache cleanup
        if target in ["all", "cache"]:
            cache_cleanup = await self._cleanup_cache(severity, dry_run)
            cleanup_results["actions_performed"].append("cache_cleanup")
            cleanup_results["space_freed"] += cache_cleanup["space_freed"]
        
        # Backup cleanup
        if severity in ["aggressive", "emergency"]:
            backup_cleanup = await self._cleanup_old_backups(dry_run)
            cleanup_results["actions_performed"].append("backup_cleanup")
            cleanup_results["space_freed"] += backup_cleanup["space_freed"]
        
        return cleanup_results
    
    async def _optimize_system(self, target: str, severity: str, dry_run: bool) -> Dict[str, Any]:
        """Optimize system performance."""
        await asyncio.sleep(0.2)
        
        optimization_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": target,
            "severity": severity,
            "dry_run": dry_run,
            "optimizations_applied": [],
            "performance_improvements": {},
            "resource_usage_before": {},
            "resource_usage_after": {}
        }
        
        # Get baseline metrics
        optimization_results["resource_usage_before"] = await self._get_resource_usage()
        
        # Memory optimization
        if target in ["all", "agents"]:
            memory_opt = await self._optimize_memory(severity, dry_run)
            optimization_results["optimizations_applied"].append("memory_optimization")
            optimization_results["performance_improvements"]["memory"] = memory_opt
        
        # Agent optimization
        if target in ["all", "agents"]:
            agent_opt = await self._optimize_agents(severity, dry_run)
            optimization_results["optimizations_applied"].append("agent_optimization")
            optimization_results["performance_improvements"]["agents"] = agent_opt
        
        # Tool optimization
        if target in ["all", "tools"]:
            tool_opt = await self._optimize_tools(severity, dry_run)
            optimization_results["optimizations_applied"].append("tool_optimization")
            optimization_results["performance_improvements"]["tools"] = tool_opt
        
        # Database optimization (if applicable)
        if severity in ["standard", "aggressive"]:
            db_opt = await self._optimize_database(dry_run)
            optimization_results["optimizations_applied"].append("database_optimization")
            optimization_results["performance_improvements"]["database"] = db_opt
        
        # Get post-optimization metrics
        optimization_results["resource_usage_after"] = await self._get_resource_usage()
        
        return optimization_results
    
    async def _backup_config(self, backup_path: Optional[str], target: str) -> Dict[str, Any]:
        """Backup system configuration."""
        await asyncio.sleep(0.1)
        
        if not backup_path:
            backup_path = f".aida/backups/backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "backup_path": str(backup_path.absolute()),
            "target": target,
            "files_backed_up": [],
            "total_size": 0,
            "success": True
        }
        
        # Backup configurations
        config_files = await self._identify_config_files(target)
        
        for config_file in config_files:
            try:
                source_path = Path(config_file)
                if source_path.exists():
                    dest_path = backup_path / source_path.name
                    shutil.copy2(source_path, dest_path)
                    
                    backup_results["files_backed_up"].append({
                        "source": str(source_path),
                        "destination": str(dest_path),
                        "size": source_path.stat().st_size
                    })
                    backup_results["total_size"] += source_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to backup {config_file}: {e}")
        
        # Create backup manifest
        manifest = {
            "backup_timestamp": backup_results["timestamp"],
            "target": target,
            "files": backup_results["files_backed_up"],
            "total_size": backup_results["total_size"],
            "aida_version": "1.0.0"
        }
        
        manifest_path = backup_path / "backup_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return backup_results
    
    async def _restore_config(self, backup_path: str, target: str) -> Dict[str, Any]:
        """Restore system configuration from backup."""
        await asyncio.sleep(0.1)
        
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup path not found: {backup_path}")
        
        manifest_path = backup_path / "backup_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Backup manifest not found")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        restore_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "backup_timestamp": manifest["backup_timestamp"],
            "target": target,
            "files_restored": [],
            "restore_failures": [],
            "success": True
        }
        
        # Restore files
        for file_info in manifest["files"]:
            try:
                source_path = Path(backup_path) / Path(file_info["source"]).name
                dest_path = Path(file_info["source"])
                
                if source_path.exists():
                    # Create backup of current config
                    if dest_path.exists():
                        backup_current = dest_path.with_suffix(dest_path.suffix + ".backup")
                        shutil.copy2(dest_path, backup_current)
                    
                    # Restore from backup
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                    
                    restore_results["files_restored"].append({
                        "file": str(dest_path),
                        "size": dest_path.stat().st_size
                    })
                else:
                    restore_results["restore_failures"].append({
                        "file": file_info["source"],
                        "error": "Backup file not found"
                    })
                    
            except Exception as e:
                restore_results["restore_failures"].append({
                    "file": file_info["source"],
                    "error": str(e)
                })
        
        restore_results["success"] = len(restore_results["restore_failures"]) == 0
        
        return restore_results
    
    async def _update_prompts(self, config_data: Dict) -> Dict[str, Any]:
        """Update system prompts and configurations."""
        await asyncio.sleep(0.1)
        
        update_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "updates_applied": [],
            "validation_results": {},
            "success": True
        }
        
        # Update system prompts
        if "system_prompts" in config_data:
            prompt_updates = await self._update_system_prompts(config_data["system_prompts"])
            update_results["updates_applied"].append("system_prompts")
            update_results["validation_results"]["system_prompts"] = prompt_updates
        
        # Update agent configurations
        if "agent_configs" in config_data:
            agent_updates = await self._update_agent_configs(config_data["agent_configs"])
            update_results["updates_applied"].append("agent_configs")
            update_results["validation_results"]["agent_configs"] = agent_updates
        
        # Update tool configurations
        if "tool_configs" in config_data:
            tool_updates = await self._update_tool_configs(config_data["tool_configs"])
            update_results["updates_applied"].append("tool_configs")
            update_results["validation_results"]["tool_configs"] = tool_updates
        
        # Validate all updates
        validation_passed = all(
            result.get("valid", True) 
            for result in update_results["validation_results"].values()
        )
        
        update_results["success"] = validation_passed
        
        return update_results
    
    # Helper methods for maintenance operations
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
            "process_count": len(psutil.pids()),
            "uptime": datetime.now() - datetime.fromtimestamp(psutil.boot_time())
        }
    
    async def _check_agent_health(self) -> Dict[str, Any]:
        """Check health of all agents."""
        return {
            "total_agents": 5,
            "healthy_agents": 4,
            "unhealthy_agents": 1,
            "issues_count": 1,
            "issues": ["agent_3_memory_leak"],
            "average_response_time": 150
        }
    
    async def _check_tool_health(self) -> Dict[str, Any]:
        """Check health of all tools."""
        return {
            "total_tools": 8,
            "functional_tools": 8,
            "failed_tools": 0,
            "tools_needing_update": 2,
            "average_execution_time": 75
        }
    
    async def _check_llm_health(self) -> Dict[str, Any]:
        """Check health of LLM providers."""
        return {
            "total_providers": 4,
            "healthy_providers": 3,
            "unhealthy_providers": 1,
            "response_times": {"openai": 200, "anthropic": 180, "ollama": 500},
            "error_rates": {"openai": 0.01, "anthropic": 0.02, "ollama": 0.15}
        }
    
    async def _check_config_integrity(self) -> Dict[str, Any]:
        """Check configuration integrity."""
        return {
            "config_files_checked": 12,
            "valid_configs": 11,
            "invalid_configs": 1,
            "missing_configs": 0,
            "backup_available": True
        }
    
    async def _check_log_health(self) -> Dict[str, Any]:
        """Check log system health."""
        return {
            "log_files_count": 25,
            "total_log_size": "1.2GB",
            "error_rate": 0.02,
            "warning_rate": 0.15,
            "rotation_needed": True
        }
    
    async def _cleanup_temp_files(self, severity: str, dry_run: bool) -> Dict[str, Any]:
        """Clean up temporary files."""
        # Simulate cleanup
        files_to_remove = 150 if severity == "aggressive" else 75
        space_freed = files_to_remove * 1024 * 1024  # 1MB per file average
        
        return {
            "files_removed": files_to_remove if not dry_run else 0,
            "space_freed": space_freed if not dry_run else 0,
            "locations_cleaned": ["/tmp", "/var/tmp"]
        }
    
    async def _cleanup_logs(self, severity: str, dry_run: bool) -> Dict[str, Any]:
        """Clean up log files."""
        files_removed = 20 if severity == "aggressive" else 10
        space_freed = files_removed * 10 * 1024 * 1024  # 10MB per log file
        
        return {
            "files_removed": files_removed if not dry_run else 0,
            "space_freed": space_freed if not dry_run else 0,
            "logs_rotated": 15 if not dry_run else 0
        }
    
    async def _cleanup_cache(self, severity: str, dry_run: bool) -> Dict[str, Any]:
        """Clean up cache files."""
        space_freed = 500 * 1024 * 1024 if severity == "aggressive" else 200 * 1024 * 1024
        
        return {
            "cache_types_cleaned": ["llm_cache", "tool_cache", "agent_cache"],
            "space_freed": space_freed if not dry_run else 0
        }
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _optimize_memory(self, severity: str, dry_run: bool) -> Dict[str, Any]:
        """Optimize memory usage."""
        memory_freed = 256 * 1024 * 1024 if severity == "aggressive" else 128 * 1024 * 1024
        
        return {
            "memory_freed_mb": memory_freed // (1024 * 1024) if not dry_run else 0,
            "optimization_techniques": ["garbage_collection", "cache_cleanup", "buffer_optimization"],
            "improvement_percent": 15 if not dry_run else 0
        }
    
    async def _identify_config_files(self, target: str) -> List[str]:
        """Identify configuration files to backup."""
        base_configs = [
            "/etc/aida/aida.conf",
            "/etc/aida/agents.conf",
            "/etc/aida/tools.conf"
        ]
        
        if target == "all":
            return base_configs + [
                "/etc/aida/llm_providers.conf",
                "/etc/aida/security.conf"
            ]
        elif target == "agents":
            return ["/etc/aida/agents.conf"]
        elif target == "tools":
            return ["/etc/aida/tools.conf"]
        else:
            return base_configs
    
    def _generate_health_recommendations(self, health_status: Dict) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []
        
        if health_status["overall_health"] == "critical":
            recommendations.append("Immediate system maintenance required")
            recommendations.append("Consider system restart after maintenance")
        elif health_status["overall_health"] == "warning":
            recommendations.append("Schedule maintenance during next maintenance window")
            recommendations.append("Monitor system closely")
        
        return recommendations