"""Architecture analysis and design tool for AIDA."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import ast
import re

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter


logger = logging.getLogger(__name__)


class ArchitectureTool(Tool):
    """Advanced architecture analysis, design, and technical requirements management tool."""
    
    def __init__(self):
        super().__init__(
            name="architecture",
            description="Analyzes technical requirements, designs system architecture, and provides architectural guidance",
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
                    description="Architecture operation to perform",
                    required=True,
                    choices=[
                        "analyze_requirements", "design_architecture", "analyze_codebase",
                        "create_diagrams", "assess_scalability", "security_review",
                        "performance_analysis", "dependency_analysis", "refactor_suggestions",
                        "migration_plan", "technology_recommendations", "architecture_patterns",
                        "api_design", "database_design", "deployment_architecture"
                    ]
                ),
                ToolParameter(
                    name="requirements",
                    type="str",
                    description="Technical requirements or specifications",
                    required=False
                ),
                ToolParameter(
                    name="codebase_path",
                    type="str",
                    description="Path to codebase for analysis",
                    required=False
                ),
                ToolParameter(
                    name="architecture_type",
                    type="str",
                    description="Type of architecture to design",
                    required=False,
                    default="microservices",
                    choices=["monolithic", "microservices", "serverless", "event_driven", "layered", "hexagonal"]
                ),
                ToolParameter(
                    name="scale_requirements",
                    type="dict",
                    description="Scalability requirements (users, requests, data)",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="technology_constraints",
                    type="list",
                    description="Technology constraints or preferences",
                    required=False,
                    default=[]
                ),
                ToolParameter(
                    name="analysis_depth",
                    type="str",
                    description="Depth of analysis to perform",
                    required=False,
                    default="standard",
                    choices=["basic", "standard", "comprehensive", "expert"]
                ),
                ToolParameter(
                    name="focus_areas",
                    type="list",
                    description="Specific areas to focus analysis on",
                    required=False,
                    choices=["performance", "security", "scalability", "maintainability", "reliability", "cost"]
                ),
                ToolParameter(
                    name="output_format",
                    type="str",
                    description="Format for architecture output",
                    required=False,
                    default="structured",
                    choices=["structured", "detailed", "visual", "summary", "technical_spec"]
                ),
                ToolParameter(
                    name="compliance_requirements",
                    type="list",
                    description="Compliance or regulatory requirements",
                    required=False,
                    default=[]
                )
            ],
            required_permissions=["file_system", "analysis"],
            supported_platforms=["any"],
            dependencies=[]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute architecture operation."""
        operation = kwargs["operation"]
        
        try:
            if operation == "analyze_requirements":
                result = await self._analyze_requirements(
                    kwargs.get("requirements", ""),
                    kwargs.get("analysis_depth", "standard"),
                    kwargs.get("focus_areas", [])
                )
            elif operation == "design_architecture":
                result = await self._design_architecture(
                    kwargs.get("requirements", ""),
                    kwargs.get("architecture_type", "microservices"),
                    kwargs.get("scale_requirements", {}),
                    kwargs.get("technology_constraints", [])
                )
            elif operation == "analyze_codebase":
                result = await self._analyze_codebase(
                    kwargs.get("codebase_path", ""),
                    kwargs.get("analysis_depth", "standard"),
                    kwargs.get("focus_areas", [])
                )
            elif operation == "create_diagrams":
                result = await self._create_diagrams(
                    kwargs.get("requirements", ""),
                    kwargs.get("architecture_type", "microservices")
                )
            elif operation == "assess_scalability":
                result = await self._assess_scalability(
                    kwargs.get("codebase_path", ""),
                    kwargs.get("scale_requirements", {}),
                    kwargs.get("architecture_type", "microservices")
                )
            elif operation == "security_review":
                result = await self._security_review(
                    kwargs.get("codebase_path", ""),
                    kwargs.get("compliance_requirements", [])
                )
            elif operation == "performance_analysis":
                result = await self._performance_analysis(
                    kwargs.get("codebase_path", ""),
                    kwargs.get("scale_requirements", {})
                )
            elif operation == "dependency_analysis":
                result = await self._dependency_analysis(
                    kwargs.get("codebase_path", "")
                )
            elif operation == "refactor_suggestions":
                result = await self._refactor_suggestions(
                    kwargs.get("codebase_path", ""),
                    kwargs.get("focus_areas", [])
                )
            elif operation == "migration_plan":
                result = await self._migration_plan(
                    kwargs.get("codebase_path", ""),
                    kwargs.get("architecture_type", "microservices"),
                    kwargs.get("requirements", "")
                )
            elif operation == "technology_recommendations":
                result = await self._technology_recommendations(
                    kwargs.get("requirements", ""),
                    kwargs.get("scale_requirements", {}),
                    kwargs.get("technology_constraints", [])
                )
            elif operation == "architecture_patterns":
                result = await self._architecture_patterns(
                    kwargs.get("requirements", ""),
                    kwargs.get("architecture_type", "microservices")
                )
            elif operation == "api_design":
                result = await self._api_design(
                    kwargs.get("requirements", ""),
                    kwargs.get("architecture_type", "microservices")
                )
            elif operation == "database_design":
                result = await self._database_design(
                    kwargs.get("requirements", ""),
                    kwargs.get("scale_requirements", {})
                )
            elif operation == "deployment_architecture":
                result = await self._deployment_architecture(
                    kwargs.get("requirements", ""),
                    kwargs.get("scale_requirements", {}),
                    kwargs.get("technology_constraints", [])
                )
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
                    "analysis_depth": kwargs.get("analysis_depth", "standard"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "focus_areas": kwargs.get("focus_areas", [])
                }
            )
            
        except Exception as e:
            raise Exception(f"Architecture operation failed: {str(e)}")
    
    async def _analyze_requirements(
        self, 
        requirements: str, 
        depth: str, 
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """Analyze technical requirements and extract architectural insights."""
        await asyncio.sleep(0.1)
        
        analysis = {
            "requirements_text": requirements,
            "analysis_depth": depth,
            "focus_areas": focus_areas,
            "extracted_requirements": {},
            "architectural_implications": {},
            "technology_suggestions": {},
            "quality_attributes": {},
            "constraints": {},
            "risks": []
        }
        
        # Extract functional requirements
        analysis["extracted_requirements"]["functional"] = self._extract_functional_requirements(requirements)
        
        # Extract non-functional requirements
        analysis["extracted_requirements"]["non_functional"] = self._extract_non_functional_requirements(requirements)
        
        # Extract business requirements
        analysis["extracted_requirements"]["business"] = self._extract_business_requirements(requirements)
        
        # Analyze architectural implications
        analysis["architectural_implications"] = self._analyze_architectural_implications(
            analysis["extracted_requirements"]
        )
        
        # Identify quality attributes
        analysis["quality_attributes"] = self._identify_quality_attributes(requirements, focus_areas)
        
        # Identify constraints
        analysis["constraints"] = self._identify_constraints(requirements)
        
        # Assess risks
        analysis["risks"] = self._assess_requirements_risks(analysis)
        
        # Generate technology suggestions
        analysis["technology_suggestions"] = self._suggest_technologies(analysis)
        
        if depth in ["comprehensive", "expert"]:
            analysis["detailed_analysis"] = self._perform_detailed_requirements_analysis(requirements)
            analysis["traceability_matrix"] = self._create_traceability_matrix(analysis)
        
        return analysis
    
    async def _design_architecture(
        self,
        requirements: str,
        architecture_type: str,
        scale_requirements: Dict,
        constraints: List[str]
    ) -> Dict[str, Any]:
        """Design system architecture based on requirements."""
        await asyncio.sleep(0.2)
        
        design = {
            "architecture_type": architecture_type,
            "requirements_summary": requirements[:200] + "..." if len(requirements) > 200 else requirements,
            "scale_requirements": scale_requirements,
            "constraints": constraints,
            "architecture_overview": {},
            "components": [],
            "interfaces": [],
            "data_flow": {},
            "deployment_view": {},
            "quality_attributes": {},
            "design_decisions": [],
            "alternatives_considered": [],
            "implementation_roadmap": {}
        }
        
        # Create architecture overview
        design["architecture_overview"] = self._create_architecture_overview(
            architecture_type, requirements, scale_requirements
        )
        
        # Design components
        design["components"] = self._design_components(architecture_type, requirements)
        
        # Design interfaces
        design["interfaces"] = self._design_interfaces(design["components"], architecture_type)
        
        # Design data flow
        design["data_flow"] = self._design_data_flow(design["components"], architecture_type)
        
        # Design deployment view
        design["deployment_view"] = self._design_deployment_view(
            design["components"], scale_requirements, constraints
        )
        
        # Address quality attributes
        design["quality_attributes"] = self._address_quality_attributes(
            architecture_type, scale_requirements
        )
        
        # Document design decisions
        design["design_decisions"] = self._document_design_decisions(
            architecture_type, requirements, constraints
        )
        
        # Consider alternatives
        design["alternatives_considered"] = self._consider_alternatives(architecture_type)
        
        # Create implementation roadmap
        design["implementation_roadmap"] = self._create_implementation_roadmap(design["components"])
        
        return design
    
    async def _analyze_codebase(
        self, 
        codebase_path: str, 
        depth: str, 
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """Analyze existing codebase architecture."""
        await asyncio.sleep(0.2)
        
        if not codebase_path or not Path(codebase_path).exists():
            raise FileNotFoundError(f"Codebase path not found: {codebase_path}")
        
        analysis = {
            "codebase_path": codebase_path,
            "analysis_depth": depth,
            "focus_areas": focus_areas,
            "codebase_metrics": {},
            "architecture_detection": {},
            "code_quality": {},
            "dependencies": {},
            "patterns_found": [],
            "anti_patterns": [],
            "recommendations": [],
            "refactoring_opportunities": []
        }
        
        # Analyze codebase metrics
        analysis["codebase_metrics"] = await self._analyze_codebase_metrics(codebase_path)
        
        # Detect current architecture
        analysis["architecture_detection"] = self._detect_current_architecture(codebase_path)
        
        # Analyze code quality
        analysis["code_quality"] = self._analyze_code_quality(codebase_path, focus_areas)
        
        # Analyze dependencies
        analysis["dependencies"] = await self._analyze_code_dependencies(codebase_path)
        
        # Identify patterns and anti-patterns
        analysis["patterns_found"] = self._identify_patterns(codebase_path)
        analysis["anti_patterns"] = self._identify_anti_patterns(codebase_path)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_architecture_recommendations(analysis)
        
        # Identify refactoring opportunities
        analysis["refactoring_opportunities"] = self._identify_refactoring_opportunities(analysis)
        
        if depth in ["comprehensive", "expert"]:
            analysis["detailed_metrics"] = await self._detailed_codebase_analysis(codebase_path)
            analysis["security_analysis"] = await self._analyze_code_security(codebase_path)
            analysis["performance_hotspots"] = self._identify_performance_hotspots(codebase_path)
        
        return analysis
    
    async def _assess_scalability(
        self,
        codebase_path: str,
        scale_requirements: Dict,
        architecture_type: str
    ) -> Dict[str, Any]:
        """Assess system scalability and provide recommendations."""
        await asyncio.sleep(0.1)
        
        assessment = {
            "scale_requirements": scale_requirements,
            "architecture_type": architecture_type,
            "scalability_analysis": {},
            "bottlenecks": [],
            "scaling_strategies": {},
            "resource_requirements": {},
            "cost_implications": {},
            "implementation_priority": []
        }
        
        # Analyze current scalability
        if codebase_path and Path(codebase_path).exists():
            assessment["current_architecture_analysis"] = self._analyze_current_scalability(codebase_path)
        
        # Identify potential bottlenecks
        assessment["bottlenecks"] = self._identify_scalability_bottlenecks(
            scale_requirements, architecture_type
        )
        
        # Suggest scaling strategies
        assessment["scaling_strategies"] = self._suggest_scaling_strategies(
            scale_requirements, architecture_type
        )
        
        # Estimate resource requirements
        assessment["resource_requirements"] = self._estimate_resource_requirements(scale_requirements)
        
        # Analyze cost implications
        assessment["cost_implications"] = self._analyze_scaling_costs(
            assessment["resource_requirements"], assessment["scaling_strategies"]
        )
        
        # Prioritize implementation
        assessment["implementation_priority"] = self._prioritize_scaling_improvements(
            assessment["bottlenecks"], assessment["scaling_strategies"]
        )
        
        return assessment
    
    async def _security_review(
        self, 
        codebase_path: str, 
        compliance_requirements: List[str]
    ) -> Dict[str, Any]:
        """Perform architectural security review."""
        await asyncio.sleep(0.1)
        
        review = {
            "codebase_path": codebase_path,
            "compliance_requirements": compliance_requirements,
            "security_assessment": {},
            "vulnerabilities": [],
            "threat_model": {},
            "security_controls": {},
            "compliance_analysis": {},
            "recommendations": []
        }
        
        # Perform security assessment
        if codebase_path and Path(codebase_path).exists():
            review["security_assessment"] = await self._assess_code_security(codebase_path)
        
        # Identify vulnerabilities
        review["vulnerabilities"] = self._identify_security_vulnerabilities(codebase_path)
        
        # Create threat model
        review["threat_model"] = self._create_threat_model(codebase_path, compliance_requirements)
        
        # Assess security controls
        review["security_controls"] = self._assess_security_controls(codebase_path)
        
        # Analyze compliance
        review["compliance_analysis"] = self._analyze_compliance(
            codebase_path, compliance_requirements
        )
        
        # Generate security recommendations
        review["recommendations"] = self._generate_security_recommendations(review)
        
        return review
    
    async def _technology_recommendations(
        self,
        requirements: str,
        scale_requirements: Dict,
        constraints: List[str]
    ) -> Dict[str, Any]:
        """Recommend technologies based on requirements."""
        await asyncio.sleep(0.1)
        
        recommendations = {
            "requirements_analysis": self._analyze_tech_requirements(requirements),
            "scale_analysis": scale_requirements,
            "constraints": constraints,
            "technology_stack": {},
            "alternatives": {},
            "decision_matrix": {},
            "implementation_considerations": {},
            "migration_path": {}
        }
        
        # Recommend technology stack
        recommendations["technology_stack"] = self._recommend_technology_stack(
            requirements, scale_requirements, constraints
        )
        
        # Provide alternatives
        recommendations["alternatives"] = self._provide_technology_alternatives(
            recommendations["technology_stack"]
        )
        
        # Create decision matrix
        recommendations["decision_matrix"] = self._create_technology_decision_matrix(
            recommendations["technology_stack"], recommendations["alternatives"]
        )
        
        # Implementation considerations
        recommendations["implementation_considerations"] = self._analyze_implementation_considerations(
            recommendations["technology_stack"]
        )
        
        # Migration path (if applicable)
        recommendations["migration_path"] = self._create_technology_migration_path(
            recommendations["technology_stack"]
        )
        
        return recommendations
    
    # Helper methods for architecture operations
    def _extract_functional_requirements(self, requirements: str) -> List[Dict[str, Any]]:
        """Extract functional requirements from text."""
        # Simplified extraction - would use NLP in real implementation
        functional_reqs = []
        
        # Look for action words and system behaviors
        action_patterns = [
            r"the system (shall|must|should|will) (.+)",
            r"users (can|may|should be able to) (.+)",
            r"the application (provides|supports|enables) (.+)"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, requirements, re.IGNORECASE)
            for match in matches:
                functional_reqs.append({
                    "type": "functional",
                    "requirement": match[1].strip(),
                    "priority": "medium",
                    "source": "extracted"
                })
        
        return functional_reqs[:10]  # Limit for demo
    
    def _extract_non_functional_requirements(self, requirements: str) -> List[Dict[str, Any]]:
        """Extract non-functional requirements."""
        non_functional_reqs = []
        
        # Performance requirements
        if any(word in requirements.lower() for word in ["performance", "speed", "latency", "response time"]):
            non_functional_reqs.append({
                "type": "performance",
                "requirement": "System must meet performance requirements",
                "priority": "high"
            })
        
        # Scalability requirements
        if any(word in requirements.lower() for word in ["scale", "users", "concurrent", "load"]):
            non_functional_reqs.append({
                "type": "scalability",
                "requirement": "System must handle specified load",
                "priority": "high"
            })
        
        # Security requirements
        if any(word in requirements.lower() for word in ["security", "authentication", "authorization", "secure"]):
            non_functional_reqs.append({
                "type": "security",
                "requirement": "System must be secure",
                "priority": "critical"
            })
        
        return non_functional_reqs
    
    def _create_architecture_overview(
        self, 
        architecture_type: str, 
        requirements: str, 
        scale_requirements: Dict
    ) -> Dict[str, Any]:
        """Create high-level architecture overview."""
        return {
            "architecture_style": architecture_type,
            "key_principles": self._get_architecture_principles(architecture_type),
            "main_components": self._identify_main_components(requirements),
            "integration_patterns": self._suggest_integration_patterns(architecture_type),
            "scalability_approach": self._define_scalability_approach(architecture_type, scale_requirements),
            "technology_alignment": self._assess_technology_alignment(architecture_type)
        }
    
    def _design_components(self, architecture_type: str, requirements: str) -> List[Dict[str, Any]]:
        """Design system components."""
        components = []
        
        if architecture_type == "microservices":
            # Extract service boundaries from requirements
            potential_services = self._identify_service_boundaries(requirements)
            
            for service in potential_services:
                components.append({
                    "name": service,
                    "type": "microservice",
                    "responsibilities": [f"Handle {service} operations"],
                    "interfaces": ["REST API", "Events"],
                    "data_store": "service_database",
                    "scalability": "horizontal"
                })
        
        elif architecture_type == "layered":
            layers = ["Presentation", "Business Logic", "Data Access", "Database"]
            
            for layer in layers:
                components.append({
                    "name": layer,
                    "type": "layer",
                    "responsibilities": [f"{layer} operations"],
                    "dependencies": [],
                    "scalability": "vertical"
                })
        
        # Add common infrastructure components
        components.extend([
            {
                "name": "API Gateway",
                "type": "infrastructure",
                "responsibilities": ["Request routing", "Authentication", "Rate limiting"],
                "interfaces": ["HTTP/HTTPS"],
                "scalability": "horizontal"
            },
            {
                "name": "Load Balancer",
                "type": "infrastructure", 
                "responsibilities": ["Traffic distribution", "Health checking"],
                "interfaces": ["TCP/HTTP"],
                "scalability": "horizontal"
            }
        ])
        
        return components
    
    def _identify_service_boundaries(self, requirements: str) -> List[str]:
        """Identify potential microservice boundaries."""
        # Simplified service identification
        services = ["user", "order", "payment", "notification", "inventory"]
        
        # Filter based on requirements content
        relevant_services = []
        for service in services:
            if service in requirements.lower():
                relevant_services.append(service)
        
        return relevant_services if relevant_services else ["core", "api", "data"]
    
    def _get_architecture_principles(self, architecture_type: str) -> List[str]:
        """Get key principles for architecture type."""
        principles = {
            "microservices": [
                "Single Responsibility",
                "Decentralized Data Management",
                "Failure Isolation",
                "DevOps Culture"
            ],
            "layered": [
                "Separation of Concerns",
                "Dependency Rule",
                "Abstraction",
                "Modularity"
            ],
            "event_driven": [
                "Loose Coupling",
                "Event Sourcing",
                "Eventual Consistency",
                "Asynchronous Processing"
            ]
        }
        
        return principles.get(architecture_type, ["Modularity", "Separation of Concerns", "Scalability"])
    
    async def _analyze_codebase_metrics(self, codebase_path: str) -> Dict[str, Any]:
        """Analyze codebase metrics."""
        path_obj = Path(codebase_path)
        
        metrics = {
            "total_files": 0,
            "total_lines": 0,
            "languages": {},
            "file_sizes": [],
            "directory_structure": {}
        }
        
        # Count files and analyze structure
        for file_path in path_obj.rglob("*"):
            if file_path.is_file():
                metrics["total_files"] += 1
                
                # Count lines
                try:
                    lines = len(file_path.read_text().splitlines())
                    metrics["total_lines"] += lines
                    metrics["file_sizes"].append(lines)
                except:
                    pass
                
                # Track languages
                suffix = file_path.suffix.lower()
                if suffix:
                    metrics["languages"][suffix] = metrics["languages"].get(suffix, 0) + 1
        
        # Calculate averages
        if metrics["file_sizes"]:
            metrics["average_file_size"] = sum(metrics["file_sizes"]) / len(metrics["file_sizes"])
            metrics["max_file_size"] = max(metrics["file_sizes"])
        
        return metrics
    
    def _detect_current_architecture(self, codebase_path: str) -> Dict[str, Any]:
        """Detect current architecture pattern."""
        detection = {
            "detected_pattern": "unknown",
            "confidence": 0.0,
            "indicators": [],
            "structure_analysis": {}
        }
        
        path_obj = Path(codebase_path)
        
        # Look for architectural indicators
        indicators = []
        
        # Check for microservices indicators
        if (path_obj / "docker-compose.yml").exists():
            indicators.append("containerization")
        if (path_obj / "kubernetes").exists():
            indicators.append("orchestration")
        
        # Check for layered architecture
        if (path_obj / "src" / "controllers").exists():
            indicators.append("mvc_pattern")
        if (path_obj / "src" / "services").exists():
            indicators.append("service_layer")
        
        detection["indicators"] = indicators
        
        # Simple pattern detection
        if "containerization" in indicators:
            detection["detected_pattern"] = "microservices"
            detection["confidence"] = 0.7
        elif "mvc_pattern" in indicators:
            detection["detected_pattern"] = "layered"
            detection["confidence"] = 0.6
        else:
            detection["detected_pattern"] = "monolithic"
            detection["confidence"] = 0.5
        
        return detection
    
    def _recommend_technology_stack(
        self, 
        requirements: str, 
        scale_requirements: Dict, 
        constraints: List[str]
    ) -> Dict[str, Any]:
        """Recommend technology stack."""
        stack = {
            "backend": {},
            "frontend": {},
            "database": {},
            "infrastructure": {},
            "monitoring": {}
        }
        
        # Backend recommendations
        if "python" not in [c.lower() for c in constraints]:
            stack["backend"] = {
                "language": "Python",
                "framework": "FastAPI",
                "rationale": "High performance, easy development, great async support"
            }
        
        # Database recommendations
        expected_users = scale_requirements.get("users", 1000)
        if expected_users > 100000:
            stack["database"] = {
                "primary": "PostgreSQL",
                "cache": "Redis",
                "search": "Elasticsearch",
                "rationale": "Handles high scale with strong consistency"
            }
        else:
            stack["database"] = {
                "primary": "PostgreSQL",
                "cache": "Redis",
                "rationale": "Reliable and feature-rich for moderate scale"
            }
        
        # Infrastructure recommendations
        if scale_requirements.get("high_availability", False):
            stack["infrastructure"] = {
                "containerization": "Docker",
                "orchestration": "Kubernetes",
                "cloud": "AWS/GCP/Azure",
                "rationale": "Ensures high availability and scalability"
            }
        
        return stack