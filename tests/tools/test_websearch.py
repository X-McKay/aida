"""Test script for web search tool."""

import asyncio
import logging

from aida.tools.websearch import SearchCategory, SearchOperation, WebSearchTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_web_search():
    """Test web search functionality."""
    logger.info("=== Testing Web Search Tool ===")

    # Initialize tool
    tool = WebSearchTool()

    try:
        # Test 1: Basic search
        logger.info("\n1. Testing basic web search...")
        result = await tool.execute(
            operation=SearchOperation.SEARCH.value,
            query="Python programming best practices",
            category=SearchCategory.GENERAL.value,
            max_results=5,
        )

        if result.status.value == "completed":
            logger.info(
                f"✅ Search successful! Found {len(result.result.get('results', []))} results"
            )
            for i, item in enumerate(result.result.get("results", [])[:3], 1):
                logger.info(f"   {i}. {item.get('title', 'No title')}")
                logger.info(f"      URL: {item.get('url', 'No URL')}")
        else:
            logger.error(f"❌ Search failed: {result.error}")

        # Test 2: Search with content scraping
        logger.info("\n2. Testing search with content scraping...")
        result = await tool.execute(
            operation=SearchOperation.SEARCH.value,
            query="OpenAI GPT-4",
            category=SearchCategory.GENERAL.value,
            max_results=3,
            scrape_content=True,
        )

        if result.status.value == "completed":
            logger.info("✅ Search with scraping successful!")
            scraped = result.result.get("details", {}).get("scraped_content", [])
            logger.info(f"   Scraped {len(scraped)} pages")
            for page in scraped[:2]:
                logger.info(
                    f"   - {page.get('title', 'No title')} ({page.get('word_count', 0)} words)"
                )
        else:
            logger.error(f"❌ Search with scraping failed: {result.error}")

        # Test 3: Image search
        logger.info("\n3. Testing image search...")
        result = await tool.execute(
            operation=SearchOperation.SEARCH.value,
            query="cute puppies",
            category=SearchCategory.IMAGES.value,
            max_results=5,
        )

        if result.status.value == "completed":
            logger.info(
                f"✅ Image search successful! Found {len(result.result.get('results', []))} images"
            )
        else:
            logger.error(f"❌ Image search failed: {result.error}")

        # Test 4: Get website content
        logger.info("\n4. Testing website content retrieval...")
        result = await tool.execute(
            operation=SearchOperation.GET_WEBSITE.value,
            url="https://www.python.org",
        )

        if result.status.value == "completed":
            content = result.result.get("website_content", {})
            logger.info("✅ Website content retrieved successfully!")
            logger.info(f"   Title: {content.get('title', 'No title')}")
            logger.info(f"   Word count: {content.get('word_count', 0)}")
            logger.info(f"   Content preview: {content.get('content', '')[:200]}...")
        else:
            logger.error(f"❌ Website content retrieval failed: {result.error}")

        # Test 5: Get current datetime
        logger.info("\n5. Testing datetime retrieval...")
        result = await tool.execute(
            operation=SearchOperation.GET_DATETIME.value,
            timezone="America/New_York",
        )

        if result.status.value == "completed":
            datetime_info = result.result.get("datetime_info", {})
            logger.info("✅ DateTime retrieved successfully!")
            logger.info(f"   Current time in New York: {datetime_info}")
        else:
            logger.error(f"❌ DateTime retrieval failed: {result.error}")

        # Test 6: Test different search categories
        logger.info("\n6. Testing different search categories...")
        categories = [
            (SearchCategory.VIDEOS, "machine learning tutorial"),
            (SearchCategory.FILES, "python cheat sheet pdf"),
            (SearchCategory.SOCIAL, "AI news"),
        ]

        for category, query in categories:
            logger.info(f"\n   Testing {category.value} search...")
            result = await tool.execute(
                operation=SearchOperation.SEARCH.value,
                query=query,
                category=category.value,
                max_results=3,
            )

            if result.status.value == "completed":
                logger.info(
                    f"   ✅ {category.value} search successful! Found {len(result.result.get('results', []))} results"
                )
            else:
                logger.error(f"   ❌ {category.value} search failed: {result.error}")

        # Test 7: Test PydanticAI compatibility
        logger.info("\n7. Testing PydanticAI compatibility...")
        pydantic_tools = tool._create_pydantic_tools()

        # Test search_web function
        results = await pydantic_tools["search_web"](
            query="artificial intelligence",
            category="general",
            max_results=3,
        )
        logger.info(f"✅ PydanticAI search_web: Found {len(results)} results")

        # Test get_current_datetime function
        datetime_result = await pydantic_tools["get_current_datetime"](timezone="UTC")
        logger.info(f"✅ PydanticAI get_current_datetime: {datetime_result}")

        logger.info("\n=== All tests completed! ===")

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        # Cleanup
        await tool.cleanup()
        logger.info("Tool cleanup completed")


async def test_error_handling():
    """Test error handling scenarios."""
    logger.info("\n=== Testing Error Handling ===")

    tool = WebSearchTool()

    try:
        # Test missing required parameters
        logger.info("\n1. Testing missing query parameter...")
        result = await tool.execute(
            operation=SearchOperation.SEARCH.value,
            # Missing query parameter
            category=SearchCategory.GENERAL.value,
        )

        if result.status.value == "failed":
            logger.info(f"✅ Correctly handled missing parameter: {result.error}")
        else:
            logger.error("❌ Should have failed with missing parameter")

        # Test invalid URL
        logger.info("\n2. Testing invalid URL...")
        result = await tool.execute(
            operation=SearchOperation.GET_WEBSITE.value,
            url="not-a-valid-url",
        )

        if result.status.value == "failed":
            logger.info(f"✅ Correctly handled invalid URL: {result.error}")
        else:
            logger.error("❌ Should have failed with invalid URL")

        # Test invalid operation
        logger.info("\n3. Testing invalid operation...")
        result = await tool.execute(
            operation="invalid_operation",
            query="test",
        )

        if result.status.value == "failed":
            logger.info(f"✅ Correctly handled invalid operation: {result.error}")
        else:
            logger.error("❌ Should have failed with invalid operation")

    except Exception as e:
        logger.error(f"Error handling test failed: {e}", exc_info=True)
    finally:
        await tool.cleanup()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_web_search())
    asyncio.run(test_error_handling())
