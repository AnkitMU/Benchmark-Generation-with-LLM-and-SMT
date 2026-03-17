from __future__ import annotations
from typing import Dict, List, Tuple
import re

OPERATORS = [
    "forAll", "exists", "select", "collect", "size", "isUnique",
    "implies", "oclIsKindOf", "oclAsType", "oclIsUndefined", "oclIsInvalid"
]


def count_operators(ocl: str) -> Dict[str, int]:
    counts: Dict[str, int] = {op: 0 for op in OPERATORS}
    for op in OPERATORS:
        counts[op] = len(re.findall(r"\b"+re.escape(op)+r"\b", ocl))
    return counts


def nav_hops(ocl: str) -> int:
    arrow_hops = ocl.count('->')
    dot_hops = len(re.findall(r"\bself\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", ocl))
    return arrow_hops + dot_hops


def quantifier_depth(ocl: str) -> int:
    return len(re.findall(r"->(?:forAll|exists|select|collect)\s*\([^|]*\|", ocl))


def types_touched(ocl: str) -> Dict[str, int]:
    # heuristic types via keywords
    real_matches = re.findall(r"\b[0-9]+\.[0-9]+\b", ocl)
    # remove reals before counting integers
    ocl_without_reals = re.sub(r"\b[0-9]+\.[0-9]+\b", " ", ocl)
    return {
        'Integer': len(re.findall(r"\b(?:0|[1-9][0-9]*)\b", ocl_without_reals)),
        'Real': len(real_matches),
        'String': len(re.findall(r"\bsize\(\)|'[^']*'\b|\"[^\"]*\"", ocl)),
        'Boolean': len(re.findall(r"\b(true|false)\b", ocl, flags=re.IGNORECASE)),
        'Enum': len(re.findall(r"::", ocl)),
    }


def compute_tc_distribution(constraints) -> Dict:
    """
    Compute Total Complexity (TC) distribution statistics.

    Returns:
        Dictionary with min, max, mean, median, stddev, and bucket counts
    """
    tc_scores = []
    for c in constraints:
        cm = getattr(c, 'metadata', {})
        if isinstance(cm, dict):
            metrics = cm.get('complexity_metrics', {})
            tc = metrics.get('tc', 0.0)
        else:
            tc = 0.0
        tc_scores.append(tc)

    if not tc_scores:
        return {}

    sorted_tc = sorted(tc_scores)
    n = len(sorted_tc)
    mean_tc = sum(sorted_tc) / n
    median_tc = sorted_tc[n // 2] if n % 2 == 1 else (sorted_tc[n // 2 - 1] + sorted_tc[n // 2]) / 2
    variance = sum((x - mean_tc) ** 2 for x in sorted_tc) / n
    stddev_tc = variance ** 0.5

    # Bucket counts
    buckets = {"trivial": 0, "easy": 0, "medium": 0, "hard": 0, "expert": 0}
    for tc in tc_scores:
        if tc <= 3.0:
            buckets["trivial"] += 1
        elif tc <= 8.0:
            buckets["easy"] += 1
        elif tc <= 16.0:
            buckets["medium"] += 1
        elif tc <= 25.0:
            buckets["hard"] += 1
        else:
            buckets["expert"] += 1

    return {
        'min': round(sorted_tc[0], 3),
        'max': round(sorted_tc[-1], 3),
        'mean': round(mean_tc, 3),
        'median': round(median_tc, 3),
        'stddev': round(stddev_tc, 3),
        'buckets': buckets,
    }


def compute_coverage(constraints) -> Dict:
    classes = set()
    attributes = set()
    associations = set()
    op_counts = {op: 0 for op in OPERATORS}
    hop_hist = {0:0, 1:0, 2:0}
    depth_hist = {0:0, 1:0, 2:0}
    type_hits = {'Integer':0,'Real':0,'String':0,'Boolean':0,'Enum':0}

    for c in constraints:
        classes.add(c.context)
        ocl = c.ocl
        # naive attr/assoc references (prefer association detection first)
        for token in re.findall(r"\bself\.[A-Za-z_][A-Za-z0-9_]*->", ocl):
            name = token.split('.')[1].split('->')[0]
            associations.add((c.context, name))
        for token in re.findall(r"\bself\.[A-Za-z_][A-Za-z0-9_]*\b", ocl):
            name = token.split('.')[1]
            if (c.context, name) not in associations:
                attributes.add((c.context, name))

        # operators
        ops = count_operators(ocl)
        for k,v in ops.items():
            op_counts[k] += v

        # hops
        hops = nav_hops(ocl)
        hop_hist[0 if hops==0 else 1 if hops==1 else 2] += 1

        # quantifier depth
        depth = quantifier_depth(ocl)
        depth_hist[0 if depth==0 else 1 if depth==1 else 2] += 1

        # types
        th = types_touched(ocl)
        for k,v in th.items():
            type_hits[k] += 1 if v>0 else 0

    result = {
        'classes_used': len(classes),
        'attributes_used': len(attributes),
        'associations_used': len(associations),
        'operator_counts': op_counts,
        'hop_hist': hop_hist,
        'depth_hist': depth_hist,
        'types': type_hits,
    }

    # Include TC distribution if constraints have complexity metrics
    tc_dist = compute_tc_distribution(constraints)
    if tc_dist:
        result['tc_distribution'] = tc_dist

    return result
