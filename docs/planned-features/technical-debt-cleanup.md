---
name: Technical Debt Cleanup
priority: medium
size: small
status: planning
owner: @lachlants
issue: N/A - Code quality maintenance
---

# Technical Debt Cleanup

**Priority:** Medium  
**Estimated Effort:** 1-2 days  
**Target Release:** Next patch version

## Phase Progress Overview

### Phase 1: Automated Style Fixes ⏳ PLANNED
**Progress:** 0% Complete  
**Current Focus:** Auto-fixable linting issues resolution

#### Subtask 1.1: Ruff Auto-fixes ⏳ PLANNED
- [ ] Run `poetry run ruff check --fix .` to resolve 646/693 auto-fixable issues
- [ ] Fix f-string placeholders (238 issues) - remove unnecessary f-strings
- [ ] Clean up trailing whitespace (159 issues)
- [ ] Modernize type annotations (UP006: Dict → dict, UP007: Union → |)
- [ ] Sort and format import blocks (42 issues)
- [ ] Add missing newlines at end of files (32 issues)

#### Subtask 1.2: Manual Style Fixes ⏳ PLANNED
- [ ] Fix loop variable usage issues (8 B007 errors)
- [ ] Address bare except clauses (2 E722 errors)
- [ ] Fix multiple statements on one line (6 E701 errors)
- [ ] Add explicit strict parameter to zip() calls (4 B905 errors)
- [ ] Resolve function redefinition (1 F811 error)

### Phase 2: Type Annotation Improvements ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** Missing return type annotations and type safety

#### Subtask 2.1: Core Module Type Fixes ⏳ PLANNED
- [ ] Add return type annotations to utility functions (utils_pipeline_yaml.py:13)
- [ ] Fix type assignment incompatibilities (source_tracking.py:71, 112)
- [ ] Add return types to error_handling.py functions (132, 163, 186, etc.)
- [ ] Fix argument type annotations in error handling functions

#### Subtask 2.2: Experimental Module Type Fixes ⏳ PLANNED
- [ ] Fix logger parameter typing in pareto.py (Optional[Logger] issue)
- [ ] Add variable type annotations (pareto.py:31, sampling.py:264)
- [ ] Fix dict indexing type issues (pareto.py:43)
- [ ] Add missing type annotations to cache-related functions

#### Subtask 2.3: CLI Module Type Fixes ⏳ PLANNED
- [ ] Fix list None type checks in run_cmd.py (lines 131, 178, 182-185)
- [ ] Add return type annotations to CLI command functions
- [ ] Fix experiment_cmd.py function type annotations (line 14, 218)
- [ ] Add type annotations to main CLI functions

### Phase 3: Enhanced Type Safety ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** Comprehensive type coverage and safety improvements

#### Subtask 3.1: Library Stub Issues ⏳ PLANNED
- [ ] Install pandas type stubs or add type ignore for pandas imports
- [ ] Handle cv2 undefined name issues (6 F821 errors)
- [ ] Review external library type coverage

#### Subtask 3.2: Advanced Type Issues ⏳ PLANNED
- [ ] Fix unreachable code statements (error_handling.py:286)
- [ ] Resolve incompatible return value types
- [ ] Address implicit Optional parameter issues
- [ ] Fix complex union type handling

---

## Current Technical Debt Status

### Linting Issues (Ruff)
**Total:** 693 errors  
**Auto-fixable:** 646 (93%)  
**Manual fixes needed:** 47 (7%)

**Top Issues by Frequency:**
- f-string without placeholders: 238 issues
- Trailing whitespace: 159 issues  
- Type annotation modernization: 140 issues
- Import formatting: 42 issues

### Type Issues (MyPy)
**Total:** 263 errors across 41 files

**Primary Categories:**
- Missing return type annotations: ~40% of issues
- Missing argument type annotations: ~30% of issues
- Type compatibility issues: ~20% of issues
- Library stub issues: ~10% of issues

## Implementation Strategy

### Recommended Order
1. **Auto-fix style issues first** - safe, mechanical changes
2. **Add missing type annotations** - improves code documentation
3. **Fix type compatibility issues** - addresses potential bugs
4. **Handle library stubs** - external dependency management

### Tools and Commands
```bash
# Phase 1: Auto-fix style issues
poetry run ruff check --fix .
poetry run ruff format .

# Phase 2: Type checking
poetry run mypy src/giflab/ --show-error-codes
poetry run mypy src/giflab/ --html-report mypy-report

# Phase 3: Validation
poetry run pytest tests/
```

### Success Metrics
- **Ruff errors**: Reduce from 693 to <50
- **MyPy errors**: Reduce from 263 to <100
- **Test coverage**: Maintain 100% test pass rate
- **Code quality**: Improve maintainability and readability

### Files Requiring Priority Attention
**High Impact:**
- `src/giflab/experimental/runner.py` - Core experimental functionality
- `src/giflab/elimination_cache.py` - Caching system
- `src/giflab/cli/experiment_cmd.py` - New preset functionality

**Medium Impact:**
- `src/giflab/error_handling.py` - Error management
- `src/giflab/source_tracking.py` - Data flow tracking
- Analysis scripts in `scripts/analysis/` - Research tools

## Risk Assessment

### Low Risk Changes
- Trailing whitespace removal
- Import sorting
- f-string placeholder fixes
- Return type annotations for void functions

### Medium Risk Changes  
- Type annotation additions to complex functions
- Union type modernization (Union → |)
- Function parameter type additions

### Higher Attention Changes
- Type compatibility fixes (assignment issues)
- Complex generic type annotations
- External library interaction types

## Dependencies and Blockers

### Prerequisites
- Current feature development complete
- All tests passing
- No pending functional changes

### Potential Blockers
- External library stub availability
- Complex generic type definitions
- Legacy code compatibility requirements

---

## Notes

This cleanup focuses on improving code quality, maintainability, and type safety without changing functionality. The technical debt has accumulated during rapid feature development and should be addressed to maintain code health for future development.

Most issues are cosmetic or related to missing type information rather than functional bugs. The automated fixes can resolve 93% of style issues, making this primarily a straightforward maintenance task.