"""Project initialization and scaffolding tool for AIDA."""

import asyncio
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import yaml

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter


logger = logging.getLogger(__name__)


class ProjectTool(Tool):
    """Project initialization, scaffolding, and structure management tool."""
    
    def __init__(self):
        super().__init__(
            name="project",
            description="Project initialization, scaffolding, dependency management, and structure creation",
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
                    description="Project operation to perform",
                    required=True,
                    choices=[
                        "init", "scaffold", "analyze_structure", "create_template",
                        "setup_dependencies", "generate_docs", "create_tests",
                        "setup_ci_cd", "validate_structure", "migrate_project",
                        "update_dependencies", "create_dockerfile", "setup_monitoring"
                    ]
                ),
                ToolParameter(
                    name="project_path",
                    type="str",
                    description="Path to the project directory",
                    required=True
                ),
                ToolParameter(
                    name="project_type",
                    type="str",
                    description="Type of project to create",
                    required=False,
                    default="python",
                    choices=["python", "javascript", "typescript", "go", "rust", "java", "microservice", "ai_agent"]
                ),
                ToolParameter(
                    name="template_name",
                    type="str",
                    description="Name of template to use",
                    required=False,
                    choices=["basic", "advanced", "microservice", "ai_agent", "web_app", "cli_tool", "library"]
                ),
                ToolParameter(
                    name="project_config",
                    type="dict",
                    description="Project configuration parameters",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="include_tests",
                    type="bool",
                    description="Include test framework setup",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="include_docs",
                    type="bool",
                    description="Include documentation setup",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="include_ci_cd",
                    type="bool",
                    description="Include CI/CD pipeline setup",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="dependencies",
                    type="list",
                    description="List of dependencies to include",
                    required=False,
                    default=[]
                ),
                ToolParameter(
                    name="license_type",
                    type="str",
                    description="License type for the project",
                    required=False,
                    default="MIT",
                    choices=["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "Proprietary"]
                ),
                ToolParameter(
                    name="git_init",
                    type="bool",
                    description="Initialize git repository",
                    required=False,
                    default=True
                )
            ],
            required_permissions=["file_system", "process_execution"],
            supported_platforms=["linux", "darwin", "windows"],
            dependencies=["git"]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute project operation."""
        operation = kwargs["operation"]
        project_path = Path(kwargs["project_path"])
        
        try:
            if operation == "init":
                result = await self._init_project(
                    project_path,
                    kwargs.get("project_type", "python"),
                    kwargs.get("project_config", {}),
                    kwargs.get("git_init", True)
                )
            elif operation == "scaffold":
                result = await self._scaffold_project(
                    project_path,
                    kwargs.get("template_name", "basic"),
                    kwargs.get("project_type", "python"),
                    kwargs.get("project_config", {}),
                    kwargs.get("include_tests", True),
                    kwargs.get("include_docs", True),
                    kwargs.get("include_ci_cd", True)
                )
            elif operation == "analyze_structure":
                result = await self._analyze_structure(project_path)
            elif operation == "create_template":
                result = await self._create_template(
                    project_path,
                    kwargs.get("template_name", "custom"),
                    kwargs.get("project_type", "python")
                )
            elif operation == "setup_dependencies":
                result = await self._setup_dependencies(
                    project_path,
                    kwargs.get("dependencies", []),
                    kwargs.get("project_type", "python")
                )
            elif operation == "generate_docs":
                result = await self._generate_docs(
                    project_path,
                    kwargs.get("project_config", {})
                )
            elif operation == "create_tests":
                result = await self._create_tests(
                    project_path,
                    kwargs.get("project_type", "python")
                )
            elif operation == "setup_ci_cd":
                result = await self._setup_ci_cd(
                    project_path,
                    kwargs.get("project_type", "python"),
                    kwargs.get("project_config", {})
                )
            elif operation == "validate_structure":
                result = await self._validate_structure(
                    project_path,
                    kwargs.get("project_type", "python")
                )
            elif operation == "migrate_project":
                result = await self._migrate_project(
                    project_path,
                    kwargs.get("project_config", {})
                )
            elif operation == "update_dependencies":
                result = await self._update_dependencies(
                    project_path,
                    kwargs.get("project_type", "python")
                )
            elif operation == "create_dockerfile":
                result = await self._create_dockerfile(
                    project_path,
                    kwargs.get("project_type", "python"),
                    kwargs.get("project_config", {})
                )
            elif operation == "setup_monitoring":
                result = await self._setup_monitoring(
                    project_path,
                    kwargs.get("project_config", {})
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
                    "project_path": str(project_path.absolute()),
                    "project_type": kwargs.get("project_type", "python"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            raise Exception(f"Project operation failed: {str(e)}")
    
    async def _init_project(
        self, 
        project_path: Path, 
        project_type: str, 
        config: Dict, 
        git_init: bool
    ) -> Dict[str, Any]:
        """Initialize a new project."""
        await asyncio.sleep(0.1)
        
        # Create project directory
        project_path.mkdir(parents=True, exist_ok=True)
        
        result = {
            "project_path": str(project_path.absolute()),
            "project_type": project_type,
            "files_created": [],
            "directories_created": [],
            "git_initialized": False,
            "success": True
        }
        
        # Create basic project structure
        basic_structure = self._get_basic_structure(project_type)
        
        for directory in basic_structure["directories"]:
            dir_path = project_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            result["directories_created"].append(str(dir_path.relative_to(project_path)))
        
        # Create basic files
        for file_info in basic_structure["files"]:
            file_path = project_path / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = self._generate_file_content(file_info, config, project_type)
            file_path.write_text(content)
            result["files_created"].append(str(file_path.relative_to(project_path)))
        
        # Initialize git repository
        if git_init:
            try:
                subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
                result["git_initialized"] = True
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to initialize git repository: {e}")
        
        return result
    
    async def _scaffold_project(
        self,
        project_path: Path,
        template_name: str,
        project_type: str,
        config: Dict,
        include_tests: bool,
        include_docs: bool,
        include_ci_cd: bool
    ) -> Dict[str, Any]:
        """Create complete project scaffold from template."""
        await asyncio.sleep(0.2)
        
        # Start with basic initialization
        init_result = await self._init_project(project_path, project_type, config, True)
        
        scaffold_result = {
            **init_result,
            "template_used": template_name,
            "features_included": [],
            "additional_files": [],
            "configuration_files": []
        }
        
        # Apply template-specific structure
        template_structure = self._get_template_structure(template_name, project_type)
        
        for file_info in template_structure["files"]:
            file_path = project_path / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = self._generate_template_content(file_info, config, project_type, template_name)
            file_path.write_text(content)
            scaffold_result["additional_files"].append(str(file_path.relative_to(project_path)))
        
        # Add tests if requested
        if include_tests:
            test_files = await self._create_test_structure(project_path, project_type)
            scaffold_result["features_included"].append("tests")
            scaffold_result["additional_files"].extend(test_files)
        
        # Add documentation if requested
        if include_docs:
            doc_files = await self._create_doc_structure(project_path, config)
            scaffold_result["features_included"].append("documentation")
            scaffold_result["additional_files"].extend(doc_files)
        
        # Add CI/CD if requested
        if include_ci_cd:
            ci_files = await self._create_ci_structure(project_path, project_type, config)
            scaffold_result["features_included"].append("ci_cd")
            scaffold_result["additional_files"].extend(ci_files)
        
        # Create project configuration file
        project_config_file = await self._create_project_config(project_path, {
            "template": template_name,
            "type": project_type,
            "features": scaffold_result["features_included"],
            "config": config
        })
        scaffold_result["configuration_files"].append(project_config_file)
        
        return scaffold_result
    
    async def _analyze_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze existing project structure."""
        await asyncio.sleep(0.1)
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")
        
        analysis = {
            "project_path": str(project_path.absolute()),
            "project_type": "unknown",
            "structure_analysis": {},
            "files_found": [],
            "directories_found": [],
            "configuration_files": [],
            "test_files": [],
            "documentation_files": [],
            "build_files": [],
            "recommendations": []
        }
        
        # Scan project structure
        for item in project_path.rglob("*"):
            if item.is_file():
                rel_path = str(item.relative_to(project_path))
                analysis["files_found"].append(rel_path)
                
                # Categorize files
                self._categorize_file(item, analysis)
            elif item.is_dir():
                rel_path = str(item.relative_to(project_path))
                analysis["directories_found"].append(rel_path)
        
        # Detect project type
        analysis["project_type"] = self._detect_project_type(project_path, analysis)
        
        # Analyze structure quality
        analysis["structure_analysis"] = self._analyze_structure_quality(analysis)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_structure_recommendations(analysis)
        
        return analysis
    
    async def _setup_dependencies(
        self, 
        project_path: Path, 
        dependencies: List[str], 
        project_type: str
    ) -> Dict[str, Any]:
        """Setup project dependencies."""
        await asyncio.sleep(0.1)
        
        result = {
            "project_path": str(project_path.absolute()),
            "project_type": project_type,
            "dependencies_requested": dependencies,
            "dependencies_installed": [],
            "dependency_files_created": [],
            "installation_errors": []
        }
        
        if project_type == "python":
            # Create requirements.txt
            req_file = project_path / "requirements.txt"
            req_content = "\n".join(dependencies) + "\n"
            req_file.write_text(req_content)
            result["dependency_files_created"].append("requirements.txt")
            
            # Create pyproject.toml if not exists
            pyproject_file = project_path / "pyproject.toml"
            if not pyproject_file.exists():
                pyproject_content = self._generate_pyproject_toml(dependencies)
                pyproject_file.write_text(pyproject_content)
                result["dependency_files_created"].append("pyproject.toml")
        
        elif project_type in ["javascript", "typescript"]:
            # Create package.json
            package_file = project_path / "package.json"
            package_content = self._generate_package_json(dependencies, project_type)
            package_file.write_text(package_content)
            result["dependency_files_created"].append("package.json")
        
        elif project_type == "go":
            # Create go.mod
            go_mod = project_path / "go.mod"
            if not go_mod.exists():
                go_mod_content = self._generate_go_mod(dependencies)
                go_mod.write_text(go_mod_content)
                result["dependency_files_created"].append("go.mod")
        
        result["dependencies_installed"] = dependencies  # Simulate installation
        
        return result
    
    async def _generate_docs(self, project_path: Path, config: Dict) -> Dict[str, Any]:
        """Generate project documentation."""
        await asyncio.sleep(0.1)
        
        docs_dir = project_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        result = {
            "docs_directory": str(docs_dir.relative_to(project_path)),
            "documentation_files": [],
            "config_used": config
        }
        
        # Create README.md
        readme_content = self._generate_readme(config)
        readme_file = project_path / "README.md"
        readme_file.write_text(readme_content)
        result["documentation_files"].append("README.md")
        
        # Create API documentation structure
        api_docs = [
            ("docs/api.md", "# API Documentation\n\nAPI documentation goes here.\n"),
            ("docs/installation.md", "# Installation Guide\n\nInstallation instructions.\n"),
            ("docs/usage.md", "# Usage Guide\n\nUsage examples and tutorials.\n"),
            ("docs/contributing.md", "# Contributing Guide\n\nContribution guidelines.\n")
        ]
        
        for doc_path, content in api_docs:
            doc_file = project_path / doc_path
            doc_file.parent.mkdir(parents=True, exist_ok=True)
            doc_file.write_text(content)
            result["documentation_files"].append(doc_path)
        
        return result
    
    async def _create_tests(self, project_path: Path, project_type: str) -> Dict[str, Any]:
        """Create test structure and files."""
        await asyncio.sleep(0.1)
        
        result = {
            "test_directory": "",
            "test_files": [],
            "test_framework": "",
            "config_files": []
        }
        
        if project_type == "python":
            test_dir = project_path / "tests"
            test_dir.mkdir(exist_ok=True)
            result["test_directory"] = str(test_dir.relative_to(project_path))
            result["test_framework"] = "pytest"
            
            # Create test files
            test_files = [
                ("tests/__init__.py", ""),
                ("tests/test_main.py", self._generate_python_test()),
                ("tests/conftest.py", self._generate_pytest_config()),
                ("pytest.ini", self._generate_pytest_ini())
            ]
            
            for test_path, content in test_files:
                test_file = project_path / test_path
                test_file.parent.mkdir(parents=True, exist_ok=True)
                test_file.write_text(content)
                result["test_files"].append(test_path)
        
        elif project_type in ["javascript", "typescript"]:
            test_dir = project_path / "__tests__"
            test_dir.mkdir(exist_ok=True)
            result["test_directory"] = str(test_dir.relative_to(project_path))
            result["test_framework"] = "jest"
            
            # Create test files
            test_files = [
                ("__tests__/main.test.js", self._generate_jest_test()),
                ("jest.config.js", self._generate_jest_config())
            ]
            
            for test_path, content in test_files:
                test_file = project_path / test_path
                test_file.parent.mkdir(parents=True, exist_ok=True)
                test_file.write_text(content)
                result["test_files"].append(test_path)
        
        return result
    
    async def _setup_ci_cd(self, project_path: Path, project_type: str, config: Dict) -> Dict[str, Any]:
        """Setup CI/CD pipeline."""
        await asyncio.sleep(0.1)
        
        result = {
            "ci_platform": "github_actions",
            "pipeline_files": [],
            "workflow_files": []
        }
        
        # Create GitHub Actions workflow
        workflows_dir = project_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Main CI workflow
        ci_workflow = self._generate_ci_workflow(project_type, config)
        ci_file = workflows_dir / "ci.yml"
        ci_file.write_text(ci_workflow)
        result["workflow_files"].append(".github/workflows/ci.yml")
        
        # Release workflow
        release_workflow = self._generate_release_workflow(project_type)
        release_file = workflows_dir / "release.yml"
        release_file.write_text(release_workflow)
        result["workflow_files"].append(".github/workflows/release.yml")
        
        return result
    
    async def _validate_structure(self, project_path: Path, project_type: str) -> Dict[str, Any]:
        """Validate project structure against best practices."""
        await asyncio.sleep(0.1)
        
        validation_result = {
            "project_path": str(project_path.absolute()),
            "project_type": project_type,
            "validation_passed": True,
            "issues_found": [],
            "recommendations": [],
            "structure_score": 0
        }
        
        # Check for required files
        required_files = self._get_required_files(project_type)
        for file_path in required_files:
            if not (project_path / file_path).exists():
                validation_result["issues_found"].append(f"Missing required file: {file_path}")
                validation_result["validation_passed"] = False
        
        # Check directory structure
        recommended_dirs = self._get_recommended_directories(project_type)
        for dir_path in recommended_dirs:
            if not (project_path / dir_path).exists():
                validation_result["recommendations"].append(f"Consider adding directory: {dir_path}")
        
        # Calculate structure score
        total_checks = len(required_files) + len(recommended_dirs)
        passed_checks = total_checks - len(validation_result["issues_found"])
        validation_result["structure_score"] = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        
        return validation_result
    
    # Helper methods for project operations
    def _get_basic_structure(self, project_type: str) -> Dict[str, Any]:
        """Get basic project structure for given type."""
        structures = {
            "python": {
                "directories": ["src", "tests", "docs"],
                "files": [
                    {"path": "README.md", "type": "readme"},
                    {"path": ".gitignore", "type": "gitignore"},
                    {"path": "requirements.txt", "type": "requirements"},
                    {"path": "src/__init__.py", "type": "init"},
                    {"path": "src/main.py", "type": "main"}
                ]
            },
            "javascript": {
                "directories": ["src", "test", "docs"],
                "files": [
                    {"path": "README.md", "type": "readme"},
                    {"path": ".gitignore", "type": "gitignore"},
                    {"path": "package.json", "type": "package"},
                    {"path": "src/index.js", "type": "main"}
                ]
            },
            "ai_agent": {
                "directories": ["agents", "tools", "providers", "tests", "docs", "configs"],
                "files": [
                    {"path": "README.md", "type": "readme"},
                    {"path": ".gitignore", "type": "gitignore"},
                    {"path": "requirements.txt", "type": "requirements"},
                    {"path": "agents/__init__.py", "type": "init"},
                    {"path": "tools/__init__.py", "type": "init"},
                    {"path": "providers/__init__.py", "type": "init"},
                    {"path": "main.py", "type": "main"}
                ]
            }
        }
        
        return structures.get(project_type, structures["python"])
    
    def _generate_file_content(self, file_info: Dict, config: Dict, project_type: str) -> str:
        """Generate content for a specific file."""
        file_type = file_info["type"]
        
        if file_type == "readme":
            return f"# {config.get('name', 'New Project')}\n\n{config.get('description', 'Project description')}\n"
        elif file_type == "gitignore":
            return self._get_gitignore_content(project_type)
        elif file_type == "requirements":
            return "# Add your Python dependencies here\n"
        elif file_type == "init":
            return '"""Package initialization."""\n'
        elif file_type == "main":
            return self._get_main_file_content(project_type)
        else:
            return ""
    
    def _get_gitignore_content(self, project_type: str) -> str:
        """Get gitignore content for project type."""
        common = """# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
"""
        
        if project_type == "python":
            return common + """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# pytest
.pytest_cache/
.coverage
htmlcov/
"""
        elif project_type in ["javascript", "typescript"]:
            return common + """
# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.eslintcache

# Build
dist/
build/
"""
        else:
            return common
    
    def _get_main_file_content(self, project_type: str) -> str:
        """Get main file content for project type."""
        if project_type == "python":
            return '''"""Main module."""

def main():
    """Main function."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''
        elif project_type == "javascript":
            return '''// Main module
console.log("Hello, World!");
'''
        elif project_type == "ai_agent":
            return '''"""AI Agent main module."""

import asyncio
from agents.base import BaseAgent

async def main():
    """Main function."""
    agent = BaseAgent()
    await agent.start()
    print("AI Agent started successfully!")

if __name__ == "__main__":
    asyncio.run(main())
'''
        else:
            return ""
    
    def _generate_readme(self, config: Dict) -> str:
        """Generate README content."""
        name = config.get("name", "Project")
        description = config.get("description", "A new project")
        
        return f"""# {name}

{description}

## Installation

```bash
# Installation instructions
```

## Usage

```bash
# Usage examples
```

## Contributing

Contributions are welcome! Please read the contributing guidelines.

## License

This project is licensed under the MIT License.
"""
    
    def _generate_python_test(self) -> str:
        """Generate Python test file."""
        return '''"""Test module."""

import pytest


def test_example():
    """Example test."""
    assert True


class TestExample:
    """Example test class."""
    
    def test_method(self):
        """Example test method."""
        assert 1 + 1 == 2
'''
    
    def _generate_ci_workflow(self, project_type: str, config: Dict) -> str:
        """Generate CI workflow."""
        if project_type == "python":
            return '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src tests/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
'''
        else:
            return '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: npm install
    
    - name: Run tests
      run: npm test
'''