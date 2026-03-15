"""
Evaluation metrics for Vietnamese lexical normalization.
- Word Accuracy: % of words correctly normalized
- CER: Character Error Rate
- ERR: Error Reduction Rate (reduction of CER relative to baseline original text)
- BLEU: corpus-level BLEU score
"""


def word_accuracy(predictions, references):
    """
    Word-level accuracy: fraction of words that match between prediction and reference.
    Averaged over all samples.
    """
    total_words = 0
    correct_words = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        # align by min length, extra words count as wrong
        max_len = max(len(pred_tokens), len(ref_tokens))
        if max_len == 0:
            continue
        for p, r in zip(pred_tokens, ref_tokens):
            if p == r:
                correct_words += 1
        total_words += max_len

    return correct_words / total_words if total_words > 0 else 0.0


def cer(predictions, references):
    """
    Character Error Rate using edit distance.
    CER = edit_distance(pred, ref) / len(ref), averaged over samples.
    """
    total_cer = 0.0
    count = 0

    for pred, ref in zip(predictions, references):
        dist = _edit_distance(pred, ref)
        total_cer += dist / max(len(ref), 1)
        count += 1

    return total_cer / count if count > 0 else 0.0


def err(predictions, references, originals):
    """
    Error Reduction Rate: (CER_baseline - CER_system) / CER_baseline.
    CER_baseline is the CER between original noisy text and reference.
    """
    baseline_cer = cer(originals, references)
    system_cer = cer(predictions, references)
    return (baseline_cer - system_cer) / baseline_cer if baseline_cer > 0 else 0.0


def bleu(predictions, references):
    """
    Corpus-level BLEU score (4-gram with smoothing).
    """
    from collections import Counter
    import math

    # collect corpus-level stats
    clipped_counts = [0] * 4
    total_counts = [0] * 4
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, 5):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)
            clipped = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            clipped_counts[n - 1] += clipped
            total_counts[n - 1] += total

    # compute BLEU with +1 smoothing
    log_bleu = 0.0
    for n in range(4):
        precision = (clipped_counts[n] + 1) / (total_counts[n] + 1)
        log_bleu += math.log(precision) / 4

    # brevity penalty
    if pred_len <= ref_len:
        bp = math.exp(1 - ref_len / max(pred_len, 1))
    else:
        bp = 1.0

    return bp * math.exp(log_bleu)


def f1_score(predictions, references):
    """
    Calculates word-level F1 score over the entire corpus.
    Precision = correct_words / total_predicted_words
    Recall = correct_words / total_reference_words
    """
    total_correct = 0
    total_pred = 0
    total_ref = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        total_pred += len(pred_tokens)
        total_ref += len(ref_tokens)
        
        # Find exact matches aligned by position
        min_len = min(len(pred_tokens), len(ref_tokens))
        for i in range(min_len):
            if pred_tokens[i] == ref_tokens[i]:
                total_correct += 1

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_ref if total_ref > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)


def compute_all_metrics(predictions, references, originals):
    """Compute all metrics, return as dict."""
    return {
        'word_accuracy': word_accuracy(predictions, references),
        'cer': cer(predictions, references),
        'err': err(predictions, references, originals),
        'bleu': bleu(predictions, references),
        'f1': f1_score(predictions, references)
    }


# --- helpers ---

def _edit_distance(s1, s2):
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def _get_ngrams(tokens, n):
    """Get n-gram counts from a list of tokens."""
    from collections import Counter
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
