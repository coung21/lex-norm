"""
Shared evaluation metrics for Vietnamese Lexical Normalization.

Metrics:
- ERR (Error Reduction Rate): Standard metric from van der Goot (2019)
- Word-level F1 / Precision / Recall for normalization actions
- Word-level accuracy
- Sentence-level accuracy (exact match)
"""


def compute_err(predictions: list[str],
                references: list[str],
                originals: list[str]) -> dict:
    """
    Compute word-level Error Reduction Rate (ERR).

    ERR = (correct_changes - incorrect_changes) / total_needed_changes

    Where (at word level):
    - total_needed_changes: # words where original != reference (gold)
    - correct_changes: # words where original != reference AND prediction == reference
    - incorrect_changes: # words where original == reference AND prediction != reference
    """
    total_needed = 0
    correct_changes = 0
    incorrect_changes = 0

    for orig, pred, ref in zip(originals, predictions, references):
        orig_tokens = orig.strip().split()
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()

        max_len = max(len(orig_tokens), len(pred_tokens), len(ref_tokens))
        orig_tokens += [""] * (max_len - len(orig_tokens))
        pred_tokens += [""] * (max_len - len(pred_tokens))
        ref_tokens += [""] * (max_len - len(ref_tokens))

        for o, p, r in zip(orig_tokens, pred_tokens, ref_tokens):
            if o != r:
                total_needed += 1
                if p == r:
                    correct_changes += 1
            else:
                if p != r:
                    incorrect_changes += 1

    err = (correct_changes - incorrect_changes) / total_needed if total_needed > 0 else 0.0

    return {
        "ERR": err,
        "total_needed_changes": total_needed,
        "correct_changes": correct_changes,
        "incorrect_changes": incorrect_changes,
    }


def compute_f1(predictions: list[str],
               references: list[str],
               originals: list[str]) -> dict:
    """
    Compute word-level Precision, Recall, and F1 for normalization actions.

    Positive = word was changed (prediction != original)
    True positive = word was changed AND the change matches the reference

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)
    """
    tp = 0
    fp = 0
    fn = 0

    for orig, pred, ref in zip(originals, predictions, references):
        orig_tokens = orig.strip().split()
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()

        max_len = max(len(orig_tokens), len(pred_tokens), len(ref_tokens))
        orig_tokens += [""] * (max_len - len(orig_tokens))
        pred_tokens += [""] * (max_len - len(pred_tokens))
        ref_tokens += [""] * (max_len - len(ref_tokens))

        for o, p, r in zip(orig_tokens, pred_tokens, ref_tokens):
            system_changed = (p != o)
            needs_change = (o != r)

            if system_changed and p == r:
                tp += 1
            elif system_changed and p != r:
                fp += 1
            elif not system_changed and needs_change:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def compute_word_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute overall word-level accuracy."""
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        max_len = max(len(pred_tokens), len(ref_tokens))
        pred_tokens += [""] * (max_len - len(pred_tokens))
        ref_tokens += [""] * (max_len - len(ref_tokens))
        for p, r in zip(pred_tokens, ref_tokens):
            total += 1
            if p == r:
                correct += 1
    return correct / total if total > 0 else 0.0


def compute_sentence_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute sentence-level accuracy (exact match)."""
    correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return correct / len(predictions) if predictions else 0.0


def compute_all_metrics(predictions: list[str],
                        references: list[str],
                        originals: list[str]) -> dict:
    """Compute all metrics and return a single dict."""
    err_metrics = compute_err(predictions, references, originals)
    f1_metrics = compute_f1(predictions, references, originals)
    word_acc = compute_word_accuracy(predictions, references)
    sent_acc = compute_sentence_accuracy(predictions, references)

    return {
        **err_metrics,
        **f1_metrics,
        "word_accuracy": word_acc,
        "sentence_accuracy": sent_acc,
        "num_samples": len(originals),
    }


def print_metrics(metrics: dict, split_name: str = ""):
    """Pretty-print metrics."""
    n = metrics.get("num_samples", "?")
    print(f"\n{'='*60}")
    print(f"  Results on {split_name} ({n} samples)")
    print(f"{'='*60}")
    print(f"  ERR:               {metrics['ERR']:.4f}  ({metrics['ERR']*100:.2f}%)")
    print(f"  F1:                {metrics['F1']:.4f}  ({metrics['F1']*100:.2f}%)")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  Word Accuracy:     {metrics['word_accuracy']:.4f}  ({metrics['word_accuracy']*100:.2f}%)")
    print(f"  Sentence Accuracy: {metrics['sentence_accuracy']:.4f}  ({metrics['sentence_accuracy']*100:.2f}%)")
    print(f"  Correct changes:   {metrics['correct_changes']}")
    print(f"  Incorrect changes: {metrics['incorrect_changes']}")
    print(f"  Total needed:      {metrics['total_needed_changes']}")
    print(f"{'='*60}")
