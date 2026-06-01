"""Run all M1 renderer tests without pytest. Exit non-zero on first failure."""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_render_models  # noqa: E402
import test_renderer  # noqa: E402

MODULES = [test_render_models, test_renderer]


def main() -> int:
    total = 0
    for mod in MODULES:
        for t in mod.TESTS:
            total += 1
            try:
                t()
                print(f"ok   {mod.__name__}.{t.__name__}")
            except Exception:
                print(f"FAIL {mod.__name__}.{t.__name__}")
                traceback.print_exc()
                return 1
    print(f"\nALL PASSED: {total} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
