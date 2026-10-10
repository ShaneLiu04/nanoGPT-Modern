"""Rule reward regressions (migrated from test_bugfixes.py)."""


def test_rule_reward_perfect_partial_and_malformed():
    """rule_reward must give full credit for perfect answers and partial credit otherwise."""
    from rewards.rule_reward import compute_reward

    # Perfect answer with derivation.
    total, fmt, proc, acc = compute_reward("Step 1: 2 + 3 = 5. <answer>5</answer>", "5")
    assert total > 1.5
    assert fmt == 0.5
    assert proc > 0.0
    assert acc == 1.2

    # Close but not exact answer.
    total, fmt, proc, acc = compute_reward("<answer>5.0001</answer>", "5")
    assert fmt == 0.5
    assert acc > 0.0

    # Malformed: multiple answer blocks.
    total, fmt, proc, acc = compute_reward("<answer>5</answer><answer>6</answer>", "5")
    assert fmt < 0.5
    assert acc == 1.2  # first block is still evaluated

    # No answer block.
    total, fmt, proc, acc = compute_reward("The answer is 5.", "5")
    assert fmt == 0.0
    assert acc == 0.0

    # Non-numeric / inf / nan should be rejected.
    for bad in ["inf", "nan", "1e1000", "foo"]:
        total, fmt, proc, acc = compute_reward(f"<answer>{bad}</answer>", "5")
        if bad == "foo":
            assert fmt < 0.5 or acc == 0.0
        else:
            assert fmt < 0.5
            assert acc == 0.0
    print("[OK] Rule reward perfect/partial/malformed cases work")


def test_rule_reward_process_score():
    """rule_reward should reward intermediate derivation steps."""
    from rewards.rule_reward import compute_reward

    _, _, proc_with, _ = compute_reward(
        "First compute 2*3 = 6, then 6+4 = 10. <answer>10</answer>", "10"
    )
    _, _, proc_without, _ = compute_reward("<answer>10</answer>", "10")
    assert proc_with > proc_without
    print("[OK] Rule reward process score works")
