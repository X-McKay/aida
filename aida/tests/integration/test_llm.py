"""LLM integration tests for refactored system."""

from typing import Dict, Any, List
from aida.tests.base import BaseTestSuite, TestResult, test_registry
from aida.llm import chat, get_llm
from aida.config.llm_profiles import Purpose


class LLMTestSuite(BaseTestSuite):
    """Test suite for LLM functionality."""
    
    def __init__(self, verbose: bool = False, persist_files: bool = False):
        super().__init__("LLM Integration", verbose, persist_files)
    
    async def test_basic_chat(self) -> Dict[str, Any]:
        """Test basic LLM chat functionality."""
        self.log("Testing basic chat with default purpose")
        
        response = await chat("What is 2+2?", Purpose.DEFAULT)
        
        if not response or len(response.strip()) < 1:
            return {"success": False, "message": "Empty or invalid response"}
        
        self.log(f"Response: {response[:100]}...")
        return {
            "success": True,
            "message": f"Got response ({len(response)} chars)",
            "response_length": len(response)
        }
    
    async def test_multiple_purposes(self) -> Dict[str, Any]:
        """Test different LLM purposes."""
        purposes_to_test = [Purpose.DEFAULT, Purpose.CODING, Purpose.REASONING, Purpose.QUICK]
        results = {}
        
        test_requests = {
            Purpose.DEFAULT: "Hello, how are you?",
            Purpose.CODING: "Write a simple Python function to add two numbers",
            Purpose.REASONING: "What are the pros and cons of remote work?", 
            Purpose.QUICK: "What's the capital of France?"
        }
        
        for purpose in purposes_to_test:
            self.log(f"Testing purpose: {purpose.value}")
            try:
                request = test_requests.get(purpose, "Test request")
                response = await chat(request, purpose)
                
                if response and len(response.strip()) > 10:
                    results[purpose.value] = {"success": True, "length": len(response)}
                    self.log(f"  ✅ {purpose.value}: {len(response)} chars")
                else:
                    results[purpose.value] = {"success": False, "reason": "Short/empty response"}
                    self.log(f"  ❌ {purpose.value}: Invalid response")
                    
            except Exception as e:
                results[purpose.value] = {"success": False, "reason": str(e)}
                self.log(f"  ❌ {purpose.value}: {str(e)}")
        
        successful_purposes = [p for p, r in results.items() if r["success"]]
        
        return {
            "success": len(successful_purposes) > 0,
            "message": f"{len(successful_purposes)}/{len(purposes_to_test)} purposes working",
            "results": results,
            "working_purposes": successful_purposes
        }
    
    async def test_manager_functionality(self) -> Dict[str, Any]:
        """Test LLM manager functionality."""
        self.log("Testing LLM manager")
        
        manager = get_llm()
        
        # Test available purposes
        purposes = manager.list_purposes()
        self.log(f"Available purposes: {[p.value for p in purposes]}")
        
        if not purposes:
            return {"success": False, "message": "No purposes available"}
        
        # Test health check
        try:
            health = await manager.health_check()
            self.log(f"Health check: {health}")
            
            healthy_count = sum(1 for status in health.values() if status)
            
            return {
                "success": len(purposes) > 0,
                "message": f"{len(purposes)} purposes, {healthy_count} healthy",
                "purposes_count": len(purposes),
                "healthy_count": healthy_count,
                "health_status": health
            }
        except Exception as e:
            return {
                "success": len(purposes) > 0,
                "message": f"{len(purposes)} purposes available (health check failed: {str(e)})",
                "purposes_count": len(purposes)
            }
    
    async def test_streaming(self) -> Dict[str, Any]:
        """Test streaming functionality."""
        self.log("Testing streaming responses")
        
        try:
            chunk_count = 0
            total_content = ""
            
            # Get the manager directly for streaming
            manager = get_llm()
            stream_generator = await manager.chat("Count from 1 to 5", Purpose.QUICK, stream=True)
            
            async for chunk in stream_generator:
                chunk_count += 1
                total_content += str(chunk)
                if chunk_count > 100:  # Safety limit
                    break
            
            return {
                "success": chunk_count > 0 and len(total_content) > 0,
                "message": f"Received {chunk_count} chunks, {len(total_content)} total chars",
                "chunk_count": chunk_count,
                "total_length": len(total_content)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Streaming failed: {str(e)}"
            }
    
    async def run_all(self) -> List[TestResult]:
        """Run all LLM tests."""
        tests = [
            ("Basic Chat", self.test_basic_chat),
            ("Multiple Purposes", self.test_multiple_purposes),
            ("Manager Functionality", self.test_manager_functionality),
            ("Streaming", self.test_streaming),
        ]
        
        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            self.results.append(result)
        
        return self.results


# Register the test suite
test_registry.register("llm", LLMTestSuite)