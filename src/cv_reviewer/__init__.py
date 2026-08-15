"""Evidence-based AI competency review from a CV.

This package never produces a hiring, pass/fail, interview, or employment decision.
"""

from cv_reviewer.reviewer import review_cv_text, review_cv_file
from cv_reviewer.schema import CompetencyReview

__all__ = ["review_cv_text", "review_cv_file", "CompetencyReview"]
__version__ = "0.1.0"
