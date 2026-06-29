"""Tests for diverse prompt template system in data/arithmetic.py."""
import os
import random


from data.arithmetic import (
    PROMPT_TEMPLATES,
    select_prompt_templates,
    format_prompt,
    generate_easy,
    generate_medium,
    generate_hard,
)


class TestSelectPromptTemplates:
    def test_diversity_zero_returns_original(self):
        templates = select_prompt_templates(0)
        assert len(templates) == 1
        assert templates[0] == PROMPT_TEMPLATES[0]

    def test_diversity_one_returns_all(self):
        templates = select_prompt_templates(1)
        assert len(templates) == len(PROMPT_TEMPLATES)
        assert templates == list(PROMPT_TEMPLATES)

    def test_diversity_n_returns_subset(self):
        templates = select_prompt_templates(3)
        assert len(templates) == 3
        assert templates == PROMPT_TEMPLATES[:3]

    def test_diversity_exceeds_total_clamped(self):
        templates = select_prompt_templates(len(PROMPT_TEMPLATES) + 5)
        assert len(templates) == len(PROMPT_TEMPLATES)


class TestFormatPrompt:
    def test_zero_diversity_uses_original(self):
        prompt = format_prompt("3 + 5", diversity=0)
        assert prompt == f"Solve: 3 + 5\nWrap your final answer in <answer>...</answer>."

    def test_all_contain_answer_tag(self):
        for t in PROMPT_TEMPLATES:
            formatted = t.format(expr="1+1")
            assert "<answer>" in formatted
            assert "</answer>" in formatted

    def test_all_contain_expr_placeholder(self):
        for t in PROMPT_TEMPLATES:
            _ = t.format(expr="2 * 3")  # should not raise KeyError

    def test_format_prompt_zero_diversity_is_deterministic(self):
        p1 = format_prompt("10 - 3", diversity=0)
        p2 = format_prompt("10 - 3", diversity=0)
        assert p1 == p2

    def test_format_prompt_with_rng_is_reproducible(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        p1 = format_prompt("7 * 8", diversity=1, rng=rng1)
        p2 = format_prompt("7 * 8", diversity=1, rng=rng2)
        assert p1 == p2


class TestGenerateWithDiversity:
    def test_easy_zero_diversity_backward_compat(self):
        data = generate_easy(10, seed=42, prompt_diversity=0)
        assert len(data) == 10
        for item in data:
            assert "Solve:" in item["prompt"]

    def test_easy_positive_diversity_varies(self):
        data = generate_easy(50, seed=42, prompt_diversity=1)
        prompts = set(item["prompt"][:20] for item in data)  # first 20 chars
        # With 12 templates, we should see more than 1 unique prefix
        assert len(prompts) > 1

    def test_medium_positive_diversity_varies(self):
        data = generate_medium(50, seed=43, prompt_diversity=1)
        prompts = set(item["prompt"][:20] for item in data)
        assert len(prompts) > 1

    def test_hard_positive_diversity_varies(self):
        data = generate_hard(50, seed=44, prompt_diversity=1)
        prompts = set(item["prompt"][:20] for item in data)
        assert len(prompts) > 1

    def test_all_levels_have_answer_field(self):
        for gen, seed in [(generate_easy, 42), (generate_medium, 43), (generate_hard, 44)]:
            data = gen(5, seed=seed, prompt_diversity=1)
            for item in data:
                assert "answer" in item
                assert "prompt" in item
                assert "level" in item

    def test_template_count(self):
        """Sanity: we have at least 10 templates."""
        assert len(PROMPT_TEMPLATES) >= 10
