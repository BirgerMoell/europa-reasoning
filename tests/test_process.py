import unittest

from europa_reasoning.models import Generation, Item, Source
from europa_reasoning.process import process_metrics, script_share, split_reasoning
from europa_reasoning.scoring import clean_boxed, last_boxed, score_item


def _item(**overrides):
    values = dict(
        id="x", suite_id="s", suite_version="0", split="dev", language="ell", script="Grek",
        template_id="t", domain="quantitative_reasoning", capability="c", task_type="numeric",
        prompt="p", gold={"value": 21}, scoring={"type": "numeric", "extract": "boxed", "tolerance": 1e-6},
        source=Source(kind="original", title="t", url="u", license="CC0-1.0"),
    )
    values.update(overrides)
    return Item(**values)


class SplitReasoningTests(unittest.TestCase):
    def test_prompt_opened_trace_is_split_at_the_closer(self):
        self.assertEqual(split_reasoning("plan\n</think>\n\nanswer", prompt_opened=True), ("plan", "answer", True))

    def test_unclosed_trace_has_no_answer(self):
        trace, answer, closed = split_reasoning(r"loop loop \boxed{4}", prompt_opened=True)
        self.assertEqual((answer, closed), ("", False))
        self.assertIn("boxed", trace)

    def test_model_opened_trace_is_detected(self):
        self.assertEqual(split_reasoning("<think>a</think>b", prompt_opened=False), ("a", "b", True))

    def test_default_mode_text_is_the_answer(self):
        self.assertEqual(split_reasoning("just text", prompt_opened=False), (None, "just text", False))


class BoxedScoringTests(unittest.TestCase):
    def test_last_boxed_handles_nested_braces(self):
        self.assertEqual(last_boxed(r"\boxed{1} then \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertIsNone(last_boxed("no box"))

    def test_clean_boxed_strips_units_and_letter_wrappers(self):
        self.assertEqual(clean_boxed(r"21\text{ euros}"), "21 euros")
        self.assertEqual(clean_boxed("(C)"), "C")
        self.assertEqual(clean_boxed(r"12\%"), "12")

    def test_numeric_item_scores_only_the_boxed_value(self):
        item = _item()
        self.assertEqual(score_item(item, r"5*4 + 3*3 = 29, so 50 - 29 = \boxed{21}").value, 1.0)
        self.assertEqual(score_item(item, r"\boxed{1 250}").value, 0.0)
        missing = score_item(item, "the change is 21 euros")
        self.assertEqual((missing.value, missing.details["boxed_found"]), (0.0, False))

    def test_choice_item_scores_boxed_letter(self):
        item = _item(task_type="multiple_choice", gold={"answer": "C"},
                     scoring={"type": "choice", "extract": "boxed"}, options=("A. x", "B. y", "C. z"))
        self.assertTrue(score_item(item, r"... therefore \boxed{C}").passed)
        self.assertTrue(score_item(item, r"\boxed{(C)}").passed)
        self.assertFalse(score_item(item, r"\boxed{B}").passed)


class ProcessMetricTests(unittest.TestCase):
    def test_script_share_counts_letters_of_the_declared_script(self):
        self.assertEqual(script_share("αβγ abc", "Grek"), 0.5)
        self.assertIsNone(script_share("123", "Grek"))

    def test_loop_and_closure_are_reported_for_reasoning_runs(self):
        loop = " ".join(["we add the two numbers again and check the result once more"] * 6)
        generation = Generation(text="", latency_ms=1.0, finish_reason="reasoning_unclosed",
                                raw={"trace_closed": False}, reasoning=loop)
        metrics = process_metrics(_item(), generation, reasoning_enabled=True)
        self.assertFalse(metrics["trace_closed"])
        self.assertTrue(metrics["trace_loop"])
        self.assertTrue(metrics["budget_exhausted"])
        self.assertFalse(metrics["answer_present"])

    def test_default_mode_flags_leaked_think_tags(self):
        generation = Generation(text="<think>x</think> 4", latency_ms=1.0)
        metrics = process_metrics(_item(), generation, reasoning_enabled=False)
        self.assertTrue(metrics["think_tag_in_answer"])
        self.assertIsNone(metrics["trace_closed"])


if __name__ == "__main__":
    unittest.main()
