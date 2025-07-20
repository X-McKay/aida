# AIDA Makefile
# Provides convenient commands for development and testing

.PHONY: help install test validate demo benchmark clean format lint type-check security docker-build docker-run

# Default target
help:
	@echo "AIDA - Advanced Intelligent Distributed Agent System"
	@echo "===================================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install       Install dependencies using uv"
	@echo "  install-dev   Install with development dependencies"
	@echo "  test          Run all tests and validations"
	@echo "  validate      Validate installation"
	@echo "  demo          Run agent interaction demo"
	@echo "  benchmark     Run performance benchmarks"
	@echo "  format        Format code with ruff"
	@echo "  lint          Run linting checks"
	@echo "  type-check    Run type checking with mypy"
	@echo "  security      Run security scans"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run with Docker Compose"
	@echo "  clean         Clean up temporary files"
	@echo "  help          Show this help message"

# Installation
install:
	@echo "📦 Installing AIDA dependencies..."
	uv sync

install-dev:
	@echo "📦 Installing AIDA with development dependencies..."
	uv sync --all-extras

# Testing and validation
test:
	@echo "🧪 Running comprehensive test suite..."
	./scripts/run_all_tests.sh

validate:
	@echo "🔍 Validating AIDA installation..."
	python3 scripts/validate_installation.py

demo:
	@echo "🎭 Running AIDA agent interaction demo..."
	python3 scripts/demo_agent_interaction.py

benchmark:
	@echo "⚡ Running AIDA performance benchmarks..."
	python3 scripts/performance_benchmark.py

# Code quality
format:
	@echo "🎨 Formatting code with ruff..."
	ruff format aida/ scripts/ tests/

lint:
	@echo "🔍 Running ruff linting..."
	ruff check aida/ scripts/ tests/ --fix

type-check:
	@echo "🔍 Running mypy type checking..."
	mypy aida/ --ignore-missing-imports

security:
	@echo "🔒 Running security scans..."
	@command -v bandit >/dev/null 2>&1 && bandit -r aida/ -f text || echo "⚠️  bandit not installed"
	@command -v safety >/dev/null 2>&1 && safety check || echo "⚠️  safety not installed"

# Docker operations
docker-build:
	@echo "🐳 Building AIDA Docker image..."
	docker build -t aida:latest .

docker-run:
	@echo "🐳 Starting AIDA with Docker Compose..."
	docker-compose up -d

docker-stop:
	@echo "🐳 Stopping AIDA Docker services..."
	docker-compose down

docker-logs:
	@echo "📋 Showing AIDA Docker logs..."
	docker-compose logs -f

# Development
dev-setup: install-dev
	@echo "🔧 Setting up development environment..."
	@command -v pre-commit >/dev/null 2>&1 && pre-commit install || echo "⚠️  pre-commit not installed"
	@echo "✅ Development environment ready!"

pre-commit:
	@echo "🔍 Running pre-commit hooks..."
	pre-commit run --all-files

# Cleanup
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Quick development workflow
quick-test: format lint
	@echo "⚡ Running quick tests..."
	python3 scripts/test_basic_functionality.py

# Full development workflow  
full-check: format lint type-check security test
	@echo "✅ Full development check complete!"

# Release preparation
release-check: clean install-dev full-check
	@echo "🚀 Release check complete!"

# Documentation
docs:
	@echo "📚 Building documentation..."
	@command -v mkdocs >/dev/null 2>&1 && mkdocs build || echo "⚠️  mkdocs not installed"

docs-serve:
	@echo "📚 Serving documentation..."
	@command -v mkdocs >/dev/null 2>&1 && mkdocs serve || echo "⚠️  mkdocs not installed"

# Monitoring and debugging
logs:
	@echo "📋 Showing AIDA logs..."
	tail -f aida/logs/*.log 2>/dev/null || echo "No log files found"

status:
	@echo "📊 AIDA system status..."
	@echo "Project: $(shell pwd)"
	@echo "Python: $(shell python3 --version)"
	@echo "Dependencies: $(shell uv --version 2>/dev/null || echo 'uv not installed')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Docker not available')"
	@echo "Git: $(shell git --version 2>/dev/null || echo 'Git not available')"

# Performance testing
perf-test:
	@echo "🏃 Running performance tests..."
	python3 scripts/performance_benchmark.py

# Load testing (if tools available)
load-test:
	@echo "🔥 Running load tests..."
	@echo "Load testing requires additional tools - implement as needed"

# CI/CD helpers
ci-test: install test

ci-build: install docker-build

ci-deploy:
	@echo "🚀 CI/CD deployment..."
	@echo "Deployment logic depends on your infrastructure"

# Database operations (if using external DB)
db-init:
	@echo "🗄️  Initializing database..."
	@echo "Database initialization depends on your setup"

db-migrate:
	@echo "🗄️  Running database migrations..."
	@echo "Migration logic depends on your database setup"

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	@echo "Backup logic depends on your data storage"

restore:
	@echo "📁 Restoring from backup..."
	@echo "Restore logic depends on your backup strategy"