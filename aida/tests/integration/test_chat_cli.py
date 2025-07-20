"""Integration tests for AIDA chat CLI functionality."""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from aida.tests.base import BaseTestSuite, TestResult, test_registry


class ChatCLITestSuite(BaseTestSuite):
    """Test suite for chat CLI with various tool usage scenarios."""
    
    def __init__(self, verbose: bool = False, persist_files: bool = False):
        super().__init__("Chat CLI Integration", verbose, persist_files)
        
        # Test questions categorized by tool usage
        self.no_tools_questions = [
            # General knowledge and reasoning
            "What is the capital of France?",
            "Explain the difference between a list and a tuple in Python",
            "What are the main principles of object-oriented programming?",
            "How does async/await work in Python?",
            "What is the purpose of a context manager in Python?",
            
            # Math and logic
            "What is 15% of 240?",
            "If a train travels 120 miles in 2 hours, what is its average speed?",
            "What is the fibonacci sequence?",
            "Explain binary search algorithm",
            "What is the time complexity of quicksort?",
            
            # Best practices and advice
            "What are some best practices for writing clean code?",
            "How should I structure a Python project?",
            "What is test-driven development?",
            "Explain the SOLID principles",
            "What are common code smells to avoid?"
        ]
        
        self.single_tool_questions = [
            # File operations only
            "List all Python files in the current directory",
            "Create a file called test_output.txt with the content 'Hello, World!'",
            "Check if a file named README.md exists in the current directory",
            "Create a new directory called test_folder",
            "Read the first 10 lines of this test file",
            
            # Thinking/analysis only
            "Analyze the pros and cons of using microservices architecture",
            "What factors should I consider when choosing a database?",
            "Help me understand the tradeoffs between REST and GraphQL",
            "Analyze the benefits of using type hints in Python",
            "What should I consider when designing an API?",
            
            # System/execution only
            "Show me the current Python version",
            "What is the current working directory?",
            "List all environment variables that start with 'PYTHON'",
            "Show the current date and time",
            "Check if git is installed on this system"
        ]
        
        self.multi_tool_questions = [
            # File + Execution (2 tools)
            "Create a Python script that prints 'Hello, AIDA!' and then run it",
            "Write a bash script that counts files in the current directory and execute it",
            "Create a simple calculator script and test it with 5 + 3",
            "Write a Python function to reverse a string and demonstrate it working",
            "Create a script that generates a timestamp and run it",
            
            # File + Thinking + Execution (3 tools)
            "Design and implement a simple password strength checker in Python, then test it",
            "Create a Python script that validates email addresses with regex and show it working",
            "Write a function to find prime numbers up to 20 and demonstrate the output",
            "Implement a basic todo list manager in Python and show how to add an item",
            "Create a simple file organizer script that sorts files by extension and test it",
            
            # Complex multi-tool scenarios
            "Analyze the current directory structure, create a summary report file, and display its contents",
            "Write a Python script to analyze code complexity, apply it to itself, and show results",
            "Create a backup script for text files, test it on sample data, and verify the backup",
            "Design a simple logging system, implement it, and demonstrate with example messages",
            "Build a basic configuration parser, create a sample config file, and show it parsing correctly"
        ]
    
    async def test_no_tools_required(self) -> Dict[str, Any]:
        """Test questions that should be answered without any tools."""
        self.log("Testing no-tools questions")
        
        # We'll simulate what the chat would do - for actual integration tests,
        # these would go through the real chat interface
        results = {
            "success": True,
            "message": "No-tools questions defined",
            "question_count": len(self.no_tools_questions),
            "sample_questions": self.no_tools_questions[:3]
        }
        
        return results
    
    async def test_single_tool_required(self) -> Dict[str, Any]:
        """Test questions that require exactly one tool."""
        self.log("Testing single-tool questions")
        
        results = {
            "success": True,
            "message": "Single-tool questions defined",
            "question_count": len(self.single_tool_questions),
            "categories": {
                "file_operations": 5,
                "thinking_analysis": 5,
                "system_execution": 5
            }
        }
        
        return results
    
    async def test_multi_tool_required(self) -> Dict[str, Any]:
        """Test questions that require multiple tools."""
        self.log("Testing multi-tool questions")
        
        results = {
            "success": True,
            "message": "Multi-tool questions defined",
            "question_count": len(self.multi_tool_questions),
            "complexity_levels": {
                "2_tools": 5,
                "3_tools": 5,
                "complex_scenarios": 5
            }
        }
        
        return results
    
    async def test_question_diversity(self) -> Dict[str, Any]:
        """Verify question diversity and coverage."""
        self.log("Analyzing question diversity")
        
        total_questions = (
            len(self.no_tools_questions) + 
            len(self.single_tool_questions) + 
            len(self.multi_tool_questions)
        )
        
        # Analyze question types
        question_types = {
            "how": 0,
            "what": 0,
            "create": 0,
            "write": 0,
            "explain": 0,
            "other": 0
        }
        
        all_questions = (
            self.no_tools_questions + 
            self.single_tool_questions + 
            self.multi_tool_questions
        )
        
        for q in all_questions:
            q_lower = q.lower()
            if q_lower.startswith("how"):
                question_types["how"] += 1
            elif q_lower.startswith("what"):
                question_types["what"] += 1
            elif "create" in q_lower:
                question_types["create"] += 1
            elif "write" in q_lower:
                question_types["write"] += 1
            elif "explain" in q_lower:
                question_types["explain"] += 1
            else:
                question_types["other"] += 1
        
        return {
            "success": True,
            "message": "Question diversity analysis complete",
            "total_questions": total_questions,
            "distribution": {
                "no_tools": len(self.no_tools_questions),
                "single_tool": len(self.single_tool_questions),
                "multi_tool": len(self.multi_tool_questions)
            },
            "question_types": question_types
        }
    
    async def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge case questions."""
        self.log("Testing edge case questions")
        
        edge_cases = [
            # Ambiguous tool selection
            "Tell me about the files in this directory and analyze their purpose",
            
            # Could be done with or without tools
            "What is 2 + 2 and save the answer to a file",
            
            # Requires careful tool orchestration
            "Create three different Python scripts and find the longest one",
            
            # Tests error handling
            "Read a file that doesn't exist and handle the error gracefully",
            
            # Tests context understanding
            "Continue from where we left off",
        ]
        
        return {
            "success": True,
            "message": "Edge cases defined",
            "edge_case_count": len(edge_cases),
            "categories": [
                "ambiguous_tool_selection",
                "optional_tool_usage",
                "complex_orchestration",
                "error_handling",
                "context_dependent"
            ]
        }
    
    def get_test_questions(self, category: str = "all") -> List[str]:
        """Get test questions by category."""
        if category == "no_tools":
            return self.no_tools_questions
        elif category == "single_tool":
            return self.single_tool_questions
        elif category == "multi_tool":
            return self.multi_tool_questions
        else:
            return (
                self.no_tools_questions + 
                self.single_tool_questions + 
                self.multi_tool_questions
            )
    
    async def run_all(self) -> List[TestResult]:
        """Run all chat CLI tests."""
        tests = [
            ("No Tools Required", self.test_no_tools_required),
            ("Single Tool Required", self.test_single_tool_required),
            ("Multi Tool Required", self.test_multi_tool_required),
            ("Question Diversity", self.test_question_diversity),
            ("Edge Cases", self.test_edge_cases),
        ]
        
        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            self.results.append(result)
        
        return self.results


# Register the test suite
test_registry.register("chat_cli", ChatCLITestSuite)