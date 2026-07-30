"""Compatibility import for the renamed batch detector module.

New code should import from :mod:`batch_detector`.
"""

from batch_detector import (  # noqa: F401
    BatchDetectionRecord,
    BatchDetector,
    BatchSummary,
    DetectionResult,
    EfficientADDetector,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
