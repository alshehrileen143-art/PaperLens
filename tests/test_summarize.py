"""Tests for summarize.py: the prompt rules and the validation of the model's JSON. OpenAI is mocked.

Run: .\\venv\\Scripts\\python.exe -m unittest tests.test_summarize -v
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import summarize  # noqa: E402
from summarize import SummarizationError, summarize_paper  # noqa: E402

VALID = {
    "objective": "Explain LangChain.", "methodology": "", "key_findings": [], "limitations": [],
    "use_cases": [{"audience": "Developers", "application": "Build apps"}],
    "technical_use_cases": ["A chatbot"],
    "ai_ml_tasks": [{"task_type": "QA", "input": "documents", "output": "answers"}],
    "data_and_tools": {"datasets": [], "variables": [], "databases": [], "tools": ["LangChain"]},
    "suggested_limitations": ["No empirical evaluation."],
}


def summarize_with(payload: dict):
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        message = SimpleNamespace(content=json.dumps(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(summarize, "get_client", return_value=client):
        return summarize_paper("Paper text."), calls


class PromptTests(unittest.TestCase):
    def test_rules(self):
        prompt = summarize.SYSTEM_PROMPT
        self.assertIn('leave it empty ("" or [])', prompt)
        self.assertIn("EVERY limitation that the authors themselves state, and ONLY those", prompt)
        self.assertIn("Never make a limitation sound stronger or weaker than the authors do", prompt)
        self.assertIn('Each must differ in MEANING from every item in "limitations"', prompt)
        self.assertIn("not tools mentioned only as examples", prompt)
        self.assertIn("never add limitations of your own here", prompt)
        self.assertIn("ONLY research variables that the study measured", prompt)
        self.assertIn('"suggested_limitations"', prompt)
        self.assertNotIn("say so briefly rather than guessing", prompt)


class ValidationTests(unittest.TestCase):
    def test_empty_paper_fields_are_valid(self):
        summary, calls = summarize_with(VALID)
        self.assertEqual(summary["limitations"], [])
        self.assertEqual(summary["methodology"], "")
        self.assertEqual(summary["suggested_limitations"], ["No empirical evaluation."])
        self.assertEqual(list(summary), list(summarize.SUMMARY_KEYS))
        self.assertIn("<paper>", calls["messages"][1]["content"])

    def test_suggested_limitations_optional(self):
        payload = {key: value for key, value in VALID.items() if key != "suggested_limitations"}
        summary, _ = summarize_with(payload)
        self.assertEqual(summary["suggested_limitations"], [])

    def test_suggested_limitations_must_be_strings(self):
        with self.assertRaises(SummarizationError):
            summarize_with({**VALID, "suggested_limitations": "one string"})

    def test_extra_keys_dropped(self):
        summary, _ = summarize_with({**VALID, "notes": "x"})
        self.assertNotIn("notes", summary)


if __name__ == "__main__":
    unittest.main()
