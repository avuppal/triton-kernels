## Summary

<!-- One paragraph describing what this PR does and why -->

## Type of Change

- [ ] Bug fix (incorrect computation, crash)
- [ ] New kernel
- [ ] Performance improvement (same correctness, better numbers)
- [ ] Documentation / benchmark table update
- [ ] Test addition / fix

## Checklist

- [ ] `pytest -m "not gpu"` passes locally
- [ ] `ruff check src/ tests/` — zero warnings
- [ ] New kernel has ≥ 5 GPU correctness test cases
- [ ] Benchmark numbers updated in README (if performance changed)
- [ ] Commit messages follow Conventional Commits

## Benchmark Results (if applicable)

> Hardware: GPU model, CUDA version, Triton version

| Shape / Size | Before | After | Delta |
|-------------|--------|-------|-------|
| | | | |

## Notes for Reviewer

<!-- Anything tricky about the implementation, known trade-offs, or follow-up work -->
