"""Metadata management for LaTeX document generation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from settings import Settings


@dataclass
class LaTeXMetadata:
    """Metadata for LaTeX document generation."""

    author: str
    title: str
    supervisor: str
    type_of_work: str
    program: str
    submission_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    digital_submission: bool = True
    faculty: str = ""
    study_program: str = ""
    company: str = ""

    @classmethod
    def from_settings(
        cls,
        generated_title: str,
        author: Optional[str] = None,
        supervisor: Optional[str] = None,
        type_of_work: Optional[str] = None,
        program: Optional[str] = None,
        submission_date: Optional[str] = None,
        digital_submission: Optional[bool] = None,
        faculty: Optional[str] = None,
        study_program: Optional[str] = None,
        company: Optional[str] = None,
    ) -> "LaTeXMetadata":
        """Create LaTeXMetadata from settings with optional parameter overrides."""
        
        # Title: use settings title if specified, otherwise use generated title
        settings_title = Settings.LATEX_TITLE
        final_title = settings_title if settings_title != "" else generated_title
        
        return cls(
            author=author if author is not None else Settings.LATEX_AUTHOR,
            title=final_title,
            supervisor=supervisor if supervisor is not None else Settings.LATEX_SUPERVISOR,
            type_of_work=type_of_work if type_of_work is not None else Settings.LATEX_TYPE_OF_WORK,
            program=program if program is not None else Settings.LATEX_PROGRAM,
            submission_date=submission_date if submission_date is not None else datetime.now().strftime("%Y-%m-%d"),
            digital_submission=digital_submission if digital_submission is not None else Settings.LATEX_DIGITAL_SUBMISSION,
            faculty=faculty if faculty is not None else Settings.LATEX_FACULTY,
            study_program=study_program if study_program is not None else Settings.LATEX_STUDY_PROGRAM,
            company=company if company is not None else Settings.LATEX_COMPANY,
        )

