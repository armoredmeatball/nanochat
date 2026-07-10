"""Focused tests for SFT best-fit packing fallbacks."""

from nanochat.dataloader import has_sft_supervised_tokens, pop_best_fit_conversation

BOS = 99  # sentinel BOS id, distinct from the content ids produced by conversation()


def conversation(length, supervised_positions=()):
    mask = [0] * length
    for position in supervised_positions:
        mask[position] = 1
    return list(range(length)), mask


def test_best_fit_is_preferred_over_truncation():
    oversized = conversation(8, supervised_positions=(6,))
    exact_fit = conversation(5, supervised_positions=(4,))
    buffer = [oversized, exact_fit]

    selected = pop_best_fit_conversation(buffer, max_length=5, bos_token=BOS, truncate_if_needed=True)

    assert selected == exact_fit
    assert buffer == [oversized]


def test_truncation_keeps_bos_and_supervised_tail():
    # render_conversation puts supervision (mask==1) at the END of a conversation,
    # so cropping must keep BOS + the tail, not the unsupervised front.
    length = 10
    ids = list(range(length))
    mask = [0] * length
    mask[8] = 1  # assistant completion near the end
    mask[9] = 1
    buffer = [(ids, mask)]

    out_ids, out_mask = pop_best_fit_conversation(buffer, max_length=5, bos_token=BOS, truncate_if_needed=True)

    assert len(out_ids) == 5 and len(out_mask) == 5
    assert out_ids[0] == BOS and out_mask[0] == 0          # BOS-aligned
    assert out_ids == [BOS] + ids[-4:]                     # keeps the tail
    assert out_mask == [0] + mask[-4:]
    assert has_sft_supervised_tokens([out_mask])           # supervision survived the crop
    assert buffer == []                                    # oversized conversation removed

    # A naive front crop would have dropped every supervised token -> all-masked row.
    assert not has_sft_supervised_tokens([mask[:5]])


def test_shortest_oversized_conversation_is_cropped_and_removed():
    longest = conversation(9, supervised_positions=(8,))
    shortest = conversation(7, supervised_positions=(4, 6))
    buffer = [longest, shortest]

    out_ids, out_mask = pop_best_fit_conversation(buffer, max_length=5, bos_token=BOS, truncate_if_needed=True)

    # Shortest oversized conversation is chosen and cropped to BOS + its 4-token tail.
    assert out_ids == [BOS] + shortest[0][-4:]
    assert out_mask == [0] + shortest[1][-4:]
    assert buffer == [longest]


def test_partial_row_does_not_truncate_oversized_conversation():
    oversized = conversation(8)
    buffer = [oversized]

    assert pop_best_fit_conversation(buffer, max_length=3, bos_token=BOS) is None
    assert buffer == [oversized]


def test_supervised_target_detection_accounts_for_shift():
    assert not has_sft_supervised_tokens([[0, 0, 0], [0, 0, 0]])
    assert not has_sft_supervised_tokens([[1, 0, 0]])
    assert has_sft_supervised_tokens([[0, 0, 1]])


def test_all_ignore_index_batch_is_nan_and_guard_rejects_it():
    # Pin the root cause: mean cross-entropy over an all-ignore_index batch is 0/0 = NaN.
    # The generator's has_sft_supervised_tokens guard flags exactly this batch (all mask
    # columns after the shift are zero) so the NaN never reaches the optimizer.
    import torch
    import torch.nn.functional as F

    logits = torch.zeros(2, 4, 8)
    targets = torch.full((2, 4), -1, dtype=torch.long)
    loss = F.cross_entropy(logits.reshape(-1, 8), targets.reshape(-1), ignore_index=-1, reduction="mean")
    assert torch.isnan(loss)

    assert not has_sft_supervised_tokens([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
