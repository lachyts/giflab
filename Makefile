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

.PHONY: data help

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

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+: .*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ": |## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$3}' 