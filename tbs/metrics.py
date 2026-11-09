"""Evaluation metrics: Kendall tau and full-list nDCG."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


def kendalls_tau(predicted_order: List[str], true_order: List[str]) -> float:
    """
    Calculate Kendall's Tau correlation coefficient between predicted and true order.
    Returns a value between -1 and 1, where 1 indicates perfect agreement.
    
    Args:
        predicted_order: The order predicted by the model
        true_order: The true/correct order (ground truth)
    
    Returns:
        Kendall's Tau score (float)
    """
    if len(predicted_order) != len(true_order):
        raise ValueError("Orders must have the same length")
    
    true_positions = {item: idx for idx, item in enumerate(true_order)}
    pred_positions = {item: idx for idx, item in enumerate(predicted_order)}
    
    if set(predicted_order) != set(true_order):
        missing = set(true_order) - set(predicted_order)
        extra = set(predicted_order) - set(true_order)
        raise ValueError(f"Orders have different items. Missing: {missing}, Extra: {extra}")
    
    n = len(true_order)
    concordant_pairs = 0
    discordant_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            item_i = true_order[i]
            item_j = true_order[j]
            
            pred_i = pred_positions[item_i]
            pred_j = pred_positions[item_j]
            
            if pred_i < pred_j:
                concordant_pairs += 1  # Agreement
            else:
                discordant_pairs += 1  # Disagreement
    
    total_pairs = n * (n - 1) / 2
    
    if total_pairs == 0:
        return 1.0  # Single item, perfect agreement
    
    tau = (concordant_pairs - discordant_pairs) / total_pairs
    return tau


def ndcg_cut_10(best_first: Sequence[str], judged: Dict[str, int]) -> float:
    """trec_eval ndcg_cut.10: linear gain, ideal ranking over every judged doc of the query."""
    dcg = sum(max(judged.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(best_first[:10]))
    ideal = sorted((r for r in judged.values() if r > 0), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_full(best_first: Sequence[str], judged: Dict[str, int]) -> float:
    """Full-list nDCG over the ranked candidates (nDCG_Q / nDCG_C): linear gain, ideal from the
    judged items among them. This is the quality metric reported in the paper."""
    gains = [max(judged.get(d, 0), 0) for d in best_first]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains, reverse=True)))
    return dcg / idcg if idcg > 0 else 0.0
