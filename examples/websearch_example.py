"""Example usage of the Web Search Tool."""

import asyncio
import logging

from aida.tools.websearch import SearchCategory, SearchOperation, WebSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO)


async def main():
    """Demonstrate web search tool usage."""
    # Initialize the tool
    tool = WebSearchTool()

    try:
        # Example 1: Basic web search
        print("=== Basic Web Search ===")
        result = await tool.execute(
            operation=SearchOperation.SEARCH,
            query="Python async programming",
            category=SearchCategory.GENERAL,
            max_results=5,
        )

        if result.status.value == "completed":
            print(f"Found {len(result.result['results'])} results:\n")
            for i, item in enumerate(result.result["results"], 1):
                print(f"{i}. {item['title']}")
                print(f"   URL: {item['url']}")
                print(f"   {item['snippet']}\n")

        # Example 2: Get current datetime
        print("\n=== Current DateTime ===")
        result = await tool.execute(
            operation=SearchOperation.GET_DATETIME,
            timezone="America/Los_Angeles",
        )

        if result.status.value == "completed":
            print(f"Current time in Los Angeles: {result.result['datetime_info']}")

        # Example 3: Search with content scraping
        print("\n=== Search with Content Scraping ===")
        result = await tool.execute(
            operation=SearchOperation.SEARCH,
            query="OpenAI GPT models",
            category=SearchCategory.GENERAL,
            max_results=3,
            scrape_content=True,
        )

        if result.status.value == "completed":
            print(f"Search returned {len(result.result['results'])} results")
            scraped = result.result.get("details", {}).get("scraped_content", [])
            if scraped:
                print(f"\nScraped content from {len(scraped)} pages:")
                for page in scraped:
                    print(f"\n- {page['title']}")
                    print(f"  Words: {page['word_count']}")
                    print(f"  Preview: {page['content'][:200]}...")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always cleanup
        await tool.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
