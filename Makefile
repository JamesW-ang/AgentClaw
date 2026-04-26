# AgentClaw v6.1 — Makefile
# 常用开发命令快捷入口

.PHONY: help test test-unit test-integration test-coverage test-coverage-ci lint format format-check clean run run-api

PYTHON = python3
PYTEST = $(PYTHON) -m pytest
COVERAGE = $(PYTHON) -m coverage
RUFF = ruff

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────
# 测试
# ──────────────────────────────────

test: ## 运行全部测试
	$(PYTEST) test/ -v --tb=short

test-unit: ## 仅单元测试（快速）
	$(PYTEST) test/ -m unit -v --tb=short

test-integration: ## 仅集成测试
	$(PYTEST) test/ -m integration -v --tb=short

test-coverage: ## 运行测试 + 生成覆盖率报告
	$(COVERAGE) run -m pytest test/ -v --tb=short
	$(COVERAGE) report -m
	$(COVERAGE) html
	@echo ""
	@echo "HTML 覆盖率报告已生成: htmlcov/index.html"

test-coverage-ci: ## CI 模式覆盖率（XML + 门禁）
	$(PYTEST) test/ \
		-v --tb=short --timeout=60 \
		--cov --cov-report=xml:coverage.xml \
		--cov-report=term-missing \
		--cov-fail-under=40

test-watch: ## 监听文件变化自动测试（需 pytest-watch）
	$(PYTEST) test/ --watch -v

# ──────────────────────────────────
# 代码质量
# ──────────────────────────────────

lint: ## Lint 检查
	$(RUFF) check . --config pyproject.toml

format: ## 自动格式化
	$(RUFF) format . --config pyproject.toml

format-check: ## 格式检查（不修改）
	$(RUFF) format --check . --config pyproject.toml

# ──────────────────────────────────
# 清理
# ──────────────────────────────────

clean: ## 清理测试产物
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage .coverage.*

# ──────────────────────────────────
# 启动
# ──────────────────────────────────

run: ## 启动 Demo UI
	$(PYTHON) demo_ui.py

run-api: ## 启动 API 服务
	$(PYTHON) main.py
