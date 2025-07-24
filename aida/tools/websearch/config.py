"""Configuration for web search tool."""

import os


class WebSearchConfig:
    """Configuration for web search operations."""

    # SearXNG server configuration
    SEARXNG_IMAGE = "overtlids/mcp-searxng-enhanced:latest"
    SEARXNG_API_BASE_URL = os.getenv("SEARXNG_ENGINE_API_BASE_URL", "http://127.0.0.1:8080/search")

    # Timezone configuration
    DEFAULT_TIMEZONE = os.getenv("DESIRED_TIMEZONE", "UTC")

    # Search limits
    MAX_SEARCH_RESULTS = 50
    DEFAULT_SEARCH_LIMIT = 10

    # Scraping configuration
    SCRAPPED_PAGES_NO = int(os.getenv("SCRAPPED_PAGES_NO", "5"))
    RETURNED_SCRAPPED_PAGES_NO = int(os.getenv("RETURNED_SCRAPPED_PAGES_NO", "3"))
    PAGE_CONTENT_WORDS_LIMIT = int(os.getenv("PAGE_CONTENT_WORDS_LIMIT", "5000"))
    CITATION_LINKS = os.getenv("CITATION_LINKS", "True").lower() == "true"

    # Category-specific limits
    MAX_IMAGE_RESULTS = int(os.getenv("MAX_IMAGE_RESULTS", "10"))
    MAX_VIDEO_RESULTS = int(os.getenv("MAX_VIDEO_RESULTS", "10"))
    MAX_FILE_RESULTS = int(os.getenv("MAX_FILE_RESULTS", "5"))
    MAX_MAP_RESULTS = int(os.getenv("MAX_MAP_RESULTS", "5"))
    MAX_SOCIAL_RESULTS = int(os.getenv("MAX_SOCIAL_RESULTS", "5"))

    # Timeout settings
    TRAFILATURA_TIMEOUT = int(os.getenv("TRAFILATURA_TIMEOUT", "15"))
    SCRAPING_TIMEOUT = int(os.getenv("SCRAPING_TIMEOUT", "20"))
    MCP_TIMEOUT = 60  # seconds

    # Caching settings
    CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "100"))
    CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "5"))
    CACHE_MAX_AGE_MINUTES = int(os.getenv("CACHE_MAX_AGE_MINUTES", "30"))

    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "10"))
    RATE_LIMIT_TIMEOUT_SECONDS = int(os.getenv("RATE_LIMIT_TIMEOUT_SECONDS", "60"))

    # Ignored websites (comma-separated)
    IGNORED_WEBSITES = os.getenv("IGNORED_WEBSITES", "").split(",")

    @classmethod
    def get_docker_args(cls, additional_env: dict[str, str] | None = None) -> list[str]:
        """Get Docker arguments for running SearXNG MCP server."""
        args = [
            "run",
            "-i",
            "--rm",
            "--network=host",
            "-e",
            f"SEARXNG_ENGINE_API_BASE_URL={cls.SEARXNG_API_BASE_URL}",
            "-e",
            f"DESIRED_TIMEZONE={cls.DEFAULT_TIMEZONE}",
            "-e",
            f"SCRAPPED_PAGES_NO={cls.SCRAPPED_PAGES_NO}",
            "-e",
            f"RETURNED_SCRAPPED_PAGES_NO={cls.RETURNED_SCRAPPED_PAGES_NO}",
            "-e",
            f"PAGE_CONTENT_WORDS_LIMIT={cls.PAGE_CONTENT_WORDS_LIMIT}",
            "-e",
            f"CITATION_LINKS={cls.CITATION_LINKS}",
            "-e",
            f"MAX_IMAGE_RESULTS={cls.MAX_IMAGE_RESULTS}",
            "-e",
            f"MAX_VIDEO_RESULTS={cls.MAX_VIDEO_RESULTS}",
            "-e",
            f"MAX_FILE_RESULTS={cls.MAX_FILE_RESULTS}",
            "-e",
            f"MAX_MAP_RESULTS={cls.MAX_MAP_RESULTS}",
            "-e",
            f"MAX_SOCIAL_RESULTS={cls.MAX_SOCIAL_RESULTS}",
            "-e",
            f"TRAFILATURA_TIMEOUT={cls.TRAFILATURA_TIMEOUT}",
            "-e",
            f"SCRAPING_TIMEOUT={cls.SCRAPING_TIMEOUT}",
            "-e",
            f"CACHE_MAXSIZE={cls.CACHE_MAXSIZE}",
            "-e",
            f"CACHE_TTL_MINUTES={cls.CACHE_TTL_MINUTES}",
            "-e",
            f"CACHE_MAX_AGE_MINUTES={cls.CACHE_MAX_AGE_MINUTES}",
            "-e",
            f"RATE_LIMIT_REQUESTS_PER_MINUTE={cls.RATE_LIMIT_REQUESTS_PER_MINUTE}",
            "-e",
            f"RATE_LIMIT_TIMEOUT_SECONDS={cls.RATE_LIMIT_TIMEOUT_SECONDS}",
        ]

        # Add ignored websites if specified
        if cls.IGNORED_WEBSITES and cls.IGNORED_WEBSITES[0]:
            args.extend(["-e", f"IGNORED_WEBSITES={','.join(cls.IGNORED_WEBSITES)}"])

        # Add any additional environment variables
        if additional_env:
            for key, value in additional_env.items():
                args.extend(["-e", f"{key}={value}"])

        # Add the image name
        args.append(cls.SEARXNG_IMAGE)

        return args
