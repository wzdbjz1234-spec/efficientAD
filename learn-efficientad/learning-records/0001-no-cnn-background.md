# Established prior knowledge: no CNN background

The user has no prior experience with CNNs. All deep learning concepts must be explained from first principles, starting with what a convolution is at the intuitive (non-mathematical) level. The user is however already running EfficientAD training successfully on a custom dataset with ORB-based ROI cropping — so they have hands-on pipeline experience but lack theoretical understanding.

**Status**: active

**Evidence**: User stated "我是一个连CNN都不熟悉的人"

**Implications**:
- Lesson 0001 must start from absolute CNN basics (what a pixel is, sliding window intuition)
- Every concept must be tied to a specific line in the codebase they already used
- Subsequent lessons should progressively connect CNN fundamentals to EfficientAD specifics
