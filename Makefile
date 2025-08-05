# GifLab project Makefile

# -----------------------------------------------------------------------------
# VARIABLES (override on command-line, e.g. `make data RAW_DIR=/path/to/raw` )
# -----------------------------------------------------------------------------

RAW_DIR ?= data/raw
CSV_DIR ?= data/csv
EDA_DIR ?= data/eda
DATE    := $(shell date +%Y%m%d_%H%M%S)
CSV_PATH := $(CSV_DIR)/results_$(DATE).csv

# -----------------------------------------------------------------------------
# TARGETS
# -----------------------------------------------------------------------------

.PHONY: data help clean-temp clean-testing-mess test-workspace test-fast test-integration test-full

data: ## Run compression pipeline on RAW_DIR and generate EDA artefacts
	@echo "🎞️  Running GifLab compression pipeline (raw=$(RAW_DIR))…"
	@echo "🔍 Validating RAW_DIR..."
	@poetry run python -c "from giflab.validation import validate_raw_dir; validate_raw_dir('$(RAW_DIR)')"
	@mkdir -p $(CSV_DIR) $(EDA_DIR)
	poetry run giflab run $(RAW_DIR) --csv $(CSV_PATH) --renders-dir data/renders --workers 0 --resume
	@echo "📊 Results CSV: $(CSV_PATH)"
	@echo "📈 Generating EDA artefacts…"
	@poetry run python -c "from giflab.eda import generate_eda; from pathlib import Path; print('   • Generating EDA into: $(EDA_DIR)'); artefacts = generate_eda(Path('$(CSV_PATH)'), Path('$(EDA_DIR)')); print(f'   • Generated {len(artefacts)} artefacts'); [print(f'     - {name}: {path}') for name, path in artefacts.items()]"
	@echo "✅ Dataset extraction + EDA complete."

test-workspace: ## Create proper test workspace structure
	@echo "🏗️  Creating test workspace structure..."
	@mkdir -p test-workspace/{manual,debug,temp,samples}
	@echo "✅ Test workspace created. See docs/guides/testing-best-practices.md for usage."

clean-temp: ## Clean temporary test files older than 7 days
	@echo "🧹 Cleaning temporary test files..."
	@find test-workspace/temp -type f -mtime +7 -delete 2>/dev/null || true
	@echo "✅ Temporary files cleaned."

clean-testing-mess: ## Emergency cleanup of testing files in root directory
	@echo "🚨 Cleaning up testing mess in root directory..."
	@echo "⚠️  This will delete all test/debug directories and files in root!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "🗑️  Removing test directories..."
	@rm -rf debug_* test_* final_* verification_* comprehensive_* deep_* gpu_* clean_test_*
	@echo "🗑️  Removing test files..."
	@rm -f *_test.gif *test*.gif debug_*.png step*.gif pipeline_*.gif
	@echo "🗑️  Removing PNG export directories..."
	@rm -rf *png_export* *png_frames* *png_sequence* *png_from_* *png_fix*
	@echo "✅ Root directory cleaned! Use 'make test-workspace' to create proper structure."

test-fast: ## Run lightning-fast test suite (<10s, development workflow)
	@echo "⚡ Running lightning-fast test suite for development..."
	@echo "⏱️  Performance monitoring: Will alert if tests exceed 10s threshold"
	@start_time=$$(date +%s); \
	export GIFLAB_ULTRA_FAST=1 GIFLAB_MAX_PIPES=3 GIFLAB_MOCK_ALL_ENGINES=1; \
	poetry run pytest -m "fast" tests/ -n auto --tb=short; \
	test_result=$$?; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "⏱️  Test execution time: $${duration}s"; \
	if [ $$duration -gt 10 ]; then \
		echo "🚨 WARNING: Fast tests took $${duration}s (exceeds 10s threshold!)"; \
		echo "💡 Consider investigating performance regression in test suite"; \
		echo "📊 Expected: ≤10s | Actual: $${duration}s | Target met: ❌"; \
	else \
		echo "✅ Performance target met: $${duration}s ≤ 10s"; \
	fi; \
	exit $$test_result

test-integration: ## Run integration test suite (<5min, pre-commit validation)  
	@echo "🔄 Running integration test suite for comprehensive validation..."
	@export GIFLAB_MAX_PIPES=10; \
	poetry run pytest -m "not slow" tests/ -n 4 --tb=short --durations=10
	@echo "✅ Integration test suite complete! Use before committing changes."

test-full: ## Run full test matrix (<30min, release validation)
	@echo "🔍 Running full test matrix for complete coverage..."
	@export GIFLAB_FULL_MATRIX=1; \
	poetry run pytest tests/ --tb=short --durations=20 --maxfail=10
	@echo "✅ Full test matrix complete! Use before major releases."

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+: .*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ": |## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$3}' 