# Name : Divyanshu Girijesh Singh
# Class : B.Tech IT B - B3
# PRN : UIT23M1128
# Assignment 2: Regular Expressions & Text Cleaning 

from pathlib import Path
import re
from typing import Any

import pandas as pd


class TextCleaner:
    """Extract contact details and common skills from resume text."""

    SKILLS = (
        "Python",
        "Java",
        "SQL",
        "Machine Learning",
        "Data Science",
        "AI",
        "NLP",
        "Deep Learning",
        "React",
        "HTML",
        "JavaScript",
    )

    @staticmethod
    def _text(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    def extract_email(self, text: Any) -> str:
        emails = re.findall(
            r"(?<![\w.-])[\w.-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])",
            self._text(text),
        )
        return ", ".join(emails)

    def extract_url(self, text: Any) -> str:
        urls = re.findall(r"https?://[^\s<>{}\"']+", self._text(text))
        return ", ".join(url.rstrip(".,;:!?)]}") for url in urls)

    def extract_phone(self, text: Any) -> str:
        candidates = re.findall(
            r"(?<!\d)\+?\d[\d\s().-]{7,}\d(?!\d)", self._text(text)
        )
        numbers = []
        for candidate in candidates:
            number = re.sub(r"\D", "", candidate)
            if 10 <= len(number) <= 15 and number not in numbers:
                numbers.append(number)
        return ", ".join(numbers)

    def extract_hashtag(self, text: Any) -> str:
        hashtags = re.findall(r"(?<!\w)#([A-Za-z0-9_]+)", self._text(text))
        return ", ".join(hashtags)

    def remove_html(self, text: Any) -> str:
        return re.sub(r"<[^>]*>", "", self._text(text))

    def remove_json(self, text: Any) -> str:
        return re.sub(r"\{[^{}]*\}", "", self._text(text))

    def clean_text(self, text: Any) -> str:
        return self.remove_json(self.remove_html(text)).strip()

    def extract_experience(self, text: Any) -> str:
        match = re.search(
            r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:\+?\s*)years?\b",
            self._text(text),
            re.IGNORECASE,
        )
        return match.group(1) if match else ""

    def extract_name(self, text: Any) -> str:
        cleaned = self.clean_text(text)
        match = re.match(
            r"\s*([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)",
            cleaned,
        )
        return " ".join(match.groups()) if match else ""

    def extract_skills(self, text: Any) -> str:
        source = self._text(text)
        found = [
            skill
            for skill in self.SKILLS
            if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", source, re.IGNORECASE)
        ]
        return ", ".join(found)


def extract_resume_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read resumes, add extracted fields, and write the resulting CSV."""
    df = pd.read_csv(input_path)
    if "resume" not in df.columns:
        raise ValueError("Input CSV must contain a 'resume' column")

    cleaner = TextCleaner()
    resume_text = df["resume"]
    df["Name"] = resume_text.apply(cleaner.extract_name)
    df["Email"] = resume_text.apply(cleaner.extract_email)
    df["Phone"] = resume_text.apply(cleaner.extract_phone)
    df["Experience"] = resume_text.apply(cleaner.extract_experience)
    df["Skills"] = resume_text.apply(cleaner.extract_skills)
    df["URL"] = resume_text.apply(cleaner.extract_url)
    df["Hashtags"] = resume_text.apply(cleaner.extract_hashtag)
    df["Clean_Text"] = resume_text.apply(cleaner.clean_text)

    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    assignment_dir = Path(__file__).resolve().parent
    input_path = assignment_dir / "resumes.csv"
    output_path = assignment_dir / "extracted_resume_data.csv"

    extracted = extract_resume_data(input_path, output_path)
    print("Original Dataset\n")
    print(pd.read_csv(input_path))
    print("\n\nExtracted Information\n")
    print(
        extracted[
            [
                "Name",
                "Email",
                "Phone",
                "Experience",
                "Skills",
                "URL",
                "Hashtags",
            ]
        ]
    )
    print(f"\nData saved successfully to {output_path}")


if __name__ == "__main__":
    main()

