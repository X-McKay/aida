"""CodingWorker agent implementation for code-related tasks.

The CodingWorker handles:
- Code analysis and quality assessment
- Code generation from specifications
- Refactoring and optimization
- Test generation
- Documentation generation
"""

import json
import logging
from pathlib import Path
from typing import Any

from aida.agents.base import (
    WorkerAgent,
    WorkerConfig,
    create_default_sandbox_config,
)
from aida.agents.mcp import FilesystemMCPTools
from aida.config.llm_profiles import Purpose
from aida.llm import get_llm

logger = logging.getLogger(__name__)


class CodingWorker(WorkerAgent):
    """Worker agent specialized for coding tasks.

    Capabilities:
    - code_analysis: Analyze code structure, quality, and metrics
    - code_generation: Generate code from specifications
    - refactoring: Improve code quality and structure
    - test_generation: Create unit tests
    - documentation: Generate code documentation
    """

    def __init__(self, worker_id: str, config: WorkerConfig | None = None):
        """Initialize the coding worker.

        Args:
            worker_id: Unique worker ID
            config: Optional configuration
        """
        # Create default config if not provided
        if config is None:
            # Create sandbox config for coding worker
            sandbox_config = create_default_sandbox_config(
                worker_type="coding",
                worker_id=worker_id,
                capabilities=[
                    "code_analysis",
                    "code_generation",
                    "refactoring",
                    "test_generation",
                    "documentation",
                ],
            )

            config = WorkerConfig(
                agent_id=worker_id,
                agent_type="coding_worker",
                capabilities=[
                    "code_analysis",
                    "code_generation",
                    "refactoring",
                    "test_generation",
                    "documentation",
                ],
                sandbox_config=sandbox_config,
                max_concurrent_tasks=3,
                allowed_mcp_servers=["filesystem"],
            )

        super().__init__(config)

        # LLM client for code generation/analysis
        self._llm_client = None

        # MCP filesystem tools
        self._fs_tools: FilesystemMCPTools | None = None

        # Code analysis cache
        self._analysis_cache: dict[str, dict[str, Any]] = {}

    async def _on_start(self) -> None:
        """Additional initialization."""
        await super()._on_start()

        # Initialize LLM client
        try:
            self._llm_client = get_llm()
            logger.info("Initialized LLM client for coding tasks")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            # Can still do some tasks without LLM

        # Initialize filesystem access
        # The base agent already initialized MCP clients based on allowed_mcp_servers
        # We just need to verify we have filesystem access
        if "filesystem" in self.mcp_clients:
            logger.info("Using MCP filesystem client from base agent")
            self._fs_client = self.mcp_clients["filesystem"]
        else:
            logger.warning("No filesystem MCP client available, file operations will be limited")
            self._fs_client = None

    async def execute_task_logic(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute coding task based on capability.

        Args:
            task_data: Task information with capability and parameters

        Returns:
            Task execution result
        """
        capability = task_data["capability"]
        parameters = task_data["parameters"]

        logger.info(f"Executing {capability} task: {task_data['description']}")

        # Route to appropriate handler
        handlers = {
            "code_analysis": self._analyze_code,
            "code_generation": self._generate_code,
            "refactoring": self._refactor_code,
            "test_generation": self._generate_tests,
            "documentation": self._generate_documentation,
        }

        handler = handlers.get(capability)
        if not handler:
            raise ValueError(f"Unknown capability: {capability}")

        # Execute with progress reporting
        task_id = task_data["task_id"]

        # Report 10% - starting
        await self._send_progress(task_id, 10, f"Starting {capability}")

        try:
            result = await handler(parameters, task_id)

            # Report 100% - complete
            await self._send_progress(task_id, 100, f"Completed {capability}")

            return result

        except Exception as e:
            logger.error(f"Error in {capability}: {e}")
            raise

    async def _analyze_code(self, params: dict[str, Any], task_id: str) -> dict[str, Any]:
        """Analyze code for quality, structure, and metrics.

        Args:
            params: Should contain 'file_path' or 'code'
            task_id: Task ID for progress reporting

        Returns:
            Analysis results
        """
        # Get code content
        if "file_path" in params:
            file_path = params["file_path"]

            # Check cache
            if file_path in self._analysis_cache:
                logger.info(f"Using cached analysis for {file_path}")
                return self._analysis_cache[file_path]

            # Read file using MCP filesystem tools
            await self._send_progress(task_id, 20, "Reading file")

            try:
                if self._fs_client:
                    # Use MCP filesystem client
                    # Convert relative path to absolute if needed
                    import os

                    abs_path = os.path.abspath(file_path)
                    logger.debug(f"Reading file via MCP: {file_path} (absolute: {abs_path})")
                    result = await self._fs_client.execute_tool("read_file", {"path": abs_path})
                    logger.debug(f"MCP result type: {type(result)}, content: {result}")

                    # Handle MCP response format: content is a list of content objects
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        if isinstance(content, list) and len(content) > 0:
                            # Extract text from first content object
                            first_item = content[0]
                            if isinstance(first_item, dict) and "text" in first_item:
                                code = first_item["text"]
                            else:
                                code = str(content)
                        else:
                            code = str(content)
                    else:
                        code = str(result)
                    logger.debug(f"Extracted code length: {len(code)}")
                else:
                    # Fallback to direct file read
                    with open(file_path) as f:
                        code = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                code = params.get("code", "")
        else:
            code = params.get("code", "")

        if not code:
            return {"error": "No code provided for analysis"}

        # Detect language
        language = self._detect_language(params.get("file_path", ""), code)

        await self._send_progress(task_id, 40, "Analyzing code structure")

        # Basic metrics
        lines = code.split("\n")
        analysis = {
            "language": language,
            "metrics": {
                "total_lines": len(lines),
                "code_lines": len(
                    [line for line in lines if line.strip() and not line.strip().startswith("#")]
                ),
                "comment_lines": len([line for line in lines if line.strip().startswith("#")]),
                "blank_lines": len([line for line in lines if not line.strip()]),
            },
        }

        # Language-specific analysis
        if language == "python":
            analysis.update(self._analyze_python(code))
        elif language in ["javascript", "typescript"]:
            analysis.update(self._analyze_javascript(code))

        await self._send_progress(task_id, 60, "Calculating complexity")

        # Calculate complexity score
        analysis["complexity_score"] = self._calculate_complexity(analysis)

        # Quality assessment
        await self._send_progress(task_id, 80, "Assessing code quality")

        if self._llm_client and params.get("detailed_analysis", False):
            quality_assessment = await self._assess_quality_with_llm(code, language)
            analysis["quality_assessment"] = quality_assessment

        # Cache if file-based
        if "file_path" in params:
            self._analysis_cache[params["file_path"]] = analysis

        return analysis

    async def _generate_code(self, params: dict[str, Any], task_id: str) -> dict[str, Any]:
        """Generate code from specifications.

        Args:
            params: Should contain 'specification', 'language', optional 'style'
            task_id: Task ID for progress

        Returns:
            Generated code
        """
        if not self._llm_client:
            return {"error": "LLM client not available for code generation"}

        specification = params.get("specification", "")
        language = params.get("language", "python")
        style = params.get("style", "clean")
        context = params.get("context", {})

        if not specification:
            return {"error": "No specification provided"}

        await self._send_progress(task_id, 30, "Preparing generation prompt")

        # Build prompt
        prompt = f"""Generate {language} code for the following specification:

Specification: {specification}

Style requirements: {style}
Context: {json.dumps(context, indent=2) if context else "None"}

Requirements:
1. Follow {language} best practices
2. Include appropriate error handling
3. Add helpful comments
4. Use type hints where applicable
5. Make the code production-ready

Generate the code:"""

        await self._send_progress(task_id, 50, "Generating code with LLM")

        try:
            generated_code = await self._llm_client.chat(prompt, purpose=Purpose.CODING)

            # Extract code from response (remove markdown if present)
            generated_code = self._extract_code_from_response(generated_code, language)

            await self._send_progress(task_id, 70, "Validating generated code")

            # Validate syntax
            validation = self._validate_syntax(generated_code, language)

            await self._send_progress(task_id, 90, "Analyzing generated code")

            # Analyze the generated code
            analysis = await self._analyze_code({"code": generated_code}, task_id)

            return {
                "generated_code": generated_code,
                "language": language,
                "validation": validation,
                "analysis": analysis,
                "specification": specification,
            }

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {"error": f"Code generation failed: {str(e)}"}

    async def _refactor_code(self, params: dict[str, Any], task_id: str) -> dict[str, Any]:
        """Refactor code for improved quality.

        Args:
            params: Should contain 'code' or 'file_path', 'objectives'
            task_id: Task ID

        Returns:
            Refactored code and changes
        """
        if not self._llm_client:
            return {"error": "LLM client not available for refactoring"}

        # Get original code
        if "file_path" in params:
            try:
                if self._fs_client:
                    result = await self._fs_client.execute_tool(
                        "read_file", {"path": params["file_path"]}
                    )
                    # Handle MCP response format
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        if (
                            isinstance(content, list)
                            and len(content) > 0
                            and isinstance(content[0], dict)
                            and "text" in content[0]
                        ):
                            original_code = content[0]["text"]
                        else:
                            original_code = str(content)
                    else:
                        original_code = str(result)
                else:
                    # Fallback to direct read
                    with open(params["file_path"]) as f:
                        original_code = f.read()
            except Exception:
                original_code = params.get("code", "")
        else:
            original_code = params.get("code", "")

        if not original_code:
            return {"error": "No code provided for refactoring"}

        objectives = params.get("objectives", ["improve readability", "reduce complexity"])
        language = self._detect_language(params.get("file_path", ""), original_code)

        await self._send_progress(task_id, 20, "Analyzing original code")

        # Analyze original
        original_analysis = await self._analyze_code({"code": original_code}, task_id)

        await self._send_progress(task_id, 40, "Generating refactoring suggestions")

        # Build refactoring prompt
        prompt = f"""Refactor this {language} code with the following objectives:
{", ".join(objectives)}

Original code:
```{language}
{original_code}
```

Current metrics:
{json.dumps(original_analysis.get("metrics", {}), indent=2)}

Provide the refactored code that improves these aspects while maintaining functionality:"""

        await self._send_progress(task_id, 60, "Applying refactoring")

        try:
            refactored_code = await self._llm_client.chat(prompt, purpose=Purpose.CODING)
            refactored_code = self._extract_code_from_response(refactored_code, language)

            await self._send_progress(task_id, 80, "Analyzing improvements")

            # Analyze refactored
            refactored_analysis = await self._analyze_code({"code": refactored_code}, task_id)

            # Calculate improvements
            improvements = self._calculate_improvements(original_analysis, refactored_analysis)

            return {
                "original_code": original_code,
                "refactored_code": refactored_code,
                "language": language,
                "improvements": improvements,
                "original_metrics": original_analysis.get("metrics", {}),
                "refactored_metrics": refactored_analysis.get("metrics", {}),
                "objectives": objectives,
            }

        except Exception as e:
            logger.error(f"Refactoring failed: {e}")
            return {"error": f"Refactoring failed: {str(e)}"}

    async def _generate_tests(self, params: dict[str, Any], task_id: str) -> dict[str, Any]:
        """Generate unit tests for code.

        Args:
            params: Should contain 'code' or 'file_path', 'framework'
            task_id: Task ID

        Returns:
            Generated tests
        """
        if not self._llm_client:
            return {"error": "LLM client not available for test generation"}

        # Get code to test
        if "file_path" in params:
            try:
                if self._fs_client:
                    result = await self._fs_client.execute_tool(
                        "read_file", {"path": params["file_path"]}
                    )
                    # Handle MCP response format
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        if (
                            isinstance(content, list)
                            and len(content) > 0
                            and isinstance(content[0], dict)
                            and "text" in content[0]
                        ):
                            code = content[0]["text"]
                        else:
                            code = str(content)
                    else:
                        code = str(result)
                else:
                    # Fallback to direct read
                    with open(params["file_path"]) as f:
                        code = f.read()
            except Exception:
                code = params.get("code", "")
        else:
            code = params.get("code", "")

        if not code:
            return {"error": "No code provided for test generation"}

        language = self._detect_language(params.get("file_path", ""), code)
        framework = params.get("framework", "pytest" if language == "python" else "jest")

        await self._send_progress(task_id, 30, "Analyzing code for test generation")

        # Build test generation prompt
        prompt = f"""Generate comprehensive unit tests for this {language} code using {framework}:

```{language}
{code}
```

Requirements:
1. Test all public functions/methods
2. Include edge cases and error conditions
3. Use appropriate assertions
4. Follow {framework} best practices
5. Aim for high code coverage

Generate the test code:"""

        await self._send_progress(task_id, 60, "Generating test cases")

        try:
            test_code = await self._llm_client.chat(prompt, purpose=Purpose.CODING)
            test_code = self._extract_code_from_response(test_code, language)

            await self._send_progress(task_id, 80, "Validating test structure")

            # Basic validation
            validation = self._validate_syntax(test_code, language)

            # Count test cases
            test_count = self._count_test_cases(test_code, language, framework)

            return {
                "test_code": test_code,
                "language": language,
                "framework": framework,
                "test_count": test_count,
                "validation": validation,
                "original_code": code,
            }

        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return {"error": f"Test generation failed: {str(e)}"}

    async def _generate_documentation(self, params: dict[str, Any], task_id: str) -> dict[str, Any]:
        """Generate documentation for code.

        Args:
            params: Should contain 'code' or 'file_path', 'format'
            task_id: Task ID

        Returns:
            Generated documentation
        """
        if not self._llm_client:
            return {"error": "LLM client not available for documentation generation"}

        # Get code
        if "file_path" in params:
            try:
                if self._fs_client:
                    result = await self._fs_client.execute_tool(
                        "read_file", {"path": params["file_path"]}
                    )
                    # Handle MCP response format
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        if (
                            isinstance(content, list)
                            and len(content) > 0
                            and isinstance(content[0], dict)
                            and "text" in content[0]
                        ):
                            code = content[0]["text"]
                        else:
                            code = str(content)
                    else:
                        code = str(result)
                else:
                    # Fallback to direct read
                    with open(params["file_path"]) as f:
                        code = f.read()
            except Exception:
                code = params.get("code", "")
        else:
            code = params.get("code", "")

        if not code:
            return {"error": "No code provided for documentation"}

        language = self._detect_language(params.get("file_path", ""), code)
        doc_format = params.get("format", "markdown")

        await self._send_progress(task_id, 30, "Analyzing code structure")

        # Build documentation prompt
        prompt = f"""Generate comprehensive documentation for this {language} code in {doc_format} format:

```{language}
{code}
```

Include:
1. Overview and purpose
2. Function/class descriptions
3. Parameter explanations
4. Return value descriptions
5. Usage examples
6. Any important notes or warnings

Generate the documentation:"""

        await self._send_progress(task_id, 60, "Generating documentation")

        try:
            documentation = await self._llm_client.chat(prompt, purpose=Purpose.CODING)

            await self._send_progress(task_id, 80, "Formatting documentation")

            # Format based on requested type
            if doc_format == "markdown":
                # Already in markdown
                formatted_doc = documentation
            elif doc_format == "docstring":
                # Convert to docstring format
                formatted_doc = self._convert_to_docstring(documentation, language)
            else:
                formatted_doc = documentation

            return {
                "documentation": formatted_doc,
                "format": doc_format,
                "language": language,
                "original_code": code,
            }

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {"error": f"Documentation generation failed: {str(e)}"}

    # Helper methods

    def _detect_language(self, file_path: str, code: str) -> str:
        """Detect programming language from file extension or code content."""
        # Check file extension first
        if file_path:
            ext_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".go": "go",
                ".rs": "rust",
                ".rb": "ruby",
                ".php": "php",
            }

            path = Path(file_path)
            if path.suffix in ext_map:
                return ext_map[path.suffix]

        # Simple heuristics from code
        if "def " in code or "import " in code:
            return "python"
        elif "function " in code or "const " in code or "let " in code:
            return "javascript"
        elif "public class" in code or "public static" in code:
            return "java"

        return "unknown"

    def _analyze_python(self, code: str) -> dict[str, Any]:
        """Python-specific code analysis."""
        import ast

        try:
            tree = ast.parse(code)

            functions = 0
            classes = 0
            imports = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions += 1
                elif isinstance(node, ast.ClassDef):
                    classes += 1
                elif isinstance(node, ast.Import | ast.ImportFrom):
                    imports += 1

            return {"structure": {"functions": functions, "classes": classes, "imports": imports}}
        except Exception:
            return {"structure": {"error": "Failed to parse Python code"}}

    def _analyze_javascript(self, code: str) -> dict[str, Any]:
        """JavaScript-specific code analysis."""
        # Simple regex-based analysis
        import re

        functions = len(re.findall(r"function\s+\w+", code))
        arrow_functions = len(re.findall(r"=>", code))
        classes = len(re.findall(r"class\s+\w+", code))
        imports = len(re.findall(r"import\s+.*from", code))

        return {
            "structure": {
                "functions": functions + arrow_functions,
                "classes": classes,
                "imports": imports,
            }
        }

    def _calculate_complexity(self, analysis: dict[str, Any]) -> int:
        """Calculate complexity score (0-100)."""
        metrics = analysis.get("metrics", {})
        structure = analysis.get("structure", {})

        # Base complexity
        complexity = 10

        # Size factor
        lines = metrics.get("code_lines", 0)
        if lines > 500:
            complexity += 30
        elif lines > 200:
            complexity += 20
        elif lines > 100:
            complexity += 10

        # Structure factor
        functions = structure.get("functions", 0)
        classes = structure.get("classes", 0)

        complexity += min(functions * 2, 30)
        complexity += min(classes * 5, 30)

        return min(complexity, 100)

    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code from LLM response, removing markdown formatting."""
        import re

        # Look for code blocks with markdown
        pattern = rf"```{language}?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Look for [PYTHON] blocks
        if language.lower() == "python":
            pattern = r"\[PYTHON\]\n(.*?)\n\[/PYTHON\]"
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()

        # Look for any language blocks
        pattern = rf"\[{language.upper()}\]\n(.*?)\n\[/{language.upper()}\]"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # If no code blocks, assume entire response is code
        return response.strip()

    def _validate_syntax(self, code: str, language: str) -> dict[str, Any]:
        """Validate code syntax."""
        if language == "python":
            import ast

            try:
                ast.parse(code)
                return {"valid": True, "errors": []}
            except SyntaxError as e:
                return {"valid": False, "errors": [str(e)]}

        # For other languages, basic validation
        return {
            "valid": True,
            "errors": [],
            "note": "Syntax validation not implemented for " + language,
        }

    def _calculate_improvements(
        self, original: dict[str, Any], refactored: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate improvements between original and refactored code."""
        orig_metrics = original.get("metrics", {})
        ref_metrics = refactored.get("metrics", {})

        improvements = {}

        # Line count reduction
        orig_lines = orig_metrics.get("code_lines", 0)
        ref_lines = ref_metrics.get("code_lines", 0)
        if orig_lines > 0:
            line_reduction = ((orig_lines - ref_lines) / orig_lines) * 100
            improvements["line_reduction_percent"] = round(line_reduction, 1)

        # Complexity reduction
        orig_complexity = original.get("complexity_score", 0)
        ref_complexity = refactored.get("complexity_score", 0)
        improvements["complexity_reduction"] = orig_complexity - ref_complexity

        return improvements

    def _count_test_cases(self, test_code: str, language: str, framework: str) -> int:
        """Count number of test cases in generated test code."""
        import re

        if language == "python" and framework == "pytest":
            # Count test functions
            return len(re.findall(r"def test_\w+", test_code))
        elif language == "javascript" and framework == "jest":
            # Count it() or test() calls
            return len(re.findall(r"(it|test)\s*\(", test_code))

        return 0

    def _convert_to_docstring(self, markdown_doc: str, language: str) -> str:
        """Convert markdown documentation to language-specific docstrings."""
        if language == "python":
            # Convert to Python docstring format
            return f'"""\n{markdown_doc}\n"""'
        elif language in ["javascript", "typescript"]:
            # Convert to JSDoc format
            lines = markdown_doc.split("\n")
            jsdoc_lines = ["/**"]
            for line in lines:
                jsdoc_lines.append(f" * {line}" if line else " *")
            jsdoc_lines.append(" */")
            return "\n".join(jsdoc_lines)

        return markdown_doc

    async def _assess_quality_with_llm(self, code: str, language: str) -> dict[str, Any]:
        """Use LLM to assess code quality."""
        prompt = f"""Assess the quality of this {language} code:

```{language}
{code}
```

Provide a brief assessment covering:
1. Code clarity and readability
2. Best practices adherence
3. Potential issues or improvements
4. Overall quality score (1-10)

Keep the assessment concise."""

        try:
            assessment = await self._llm_client.chat(prompt, purpose=Purpose.CODING)
            return {"assessment": assessment}
        except Exception as e:
            return {"assessment": f"Quality assessment failed: {str(e)}"}
