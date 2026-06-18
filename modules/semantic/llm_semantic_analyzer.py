"""
LLM-based Semantic Attribute Analyzer using Phi-4 via Ollama.

Replaces hardcoded keyword heuristics with LLM-powered semantic grouping.
Runs ONCE per metamodel (pre-generation phase), builds a compatibility matrix,
and caches the result to JSON for reuse.

Architecture:
    Phase 1 (pre-generation):
      - Filter: skip classes with <3 same-type attrs (heuristics handle them)
      - Smart batch: group classes by attribute-pair count (max ~30 pairs/batch)
      - Stream: call Ollama with stream=True (never times out)
      - Parse: extract JSON per batch → merge into full matrix
    Phase 2 (generation):
      - O(1) dict lookup per attribute pair

Usage:
    analyzer = LLMSemanticAnalyzer()
    matrix = analyzer.analyze_metamodel(metamodel)
    matrix.is_comparable("Sensor", "value", "threshold")  # True
    matrix.is_comparable("Sensor", "age", "cost")          # False
"""
from __future__ import annotations

import json
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
# DEFAULT_MODEL = "phi4"              # 14B — more accurate
DEFAULT_MODEL = "phi4-mini"            # 3.8B — lighter, faster
CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "semantic"

# Smart batching: max attribute pairs per LLM call
# phi4-mini needs ~100 output tokens per class (JSON structure + pairs)
# 5 classes × ~100 tokens = ~500 tokens → well within limits
MAX_PAIRS_PER_BATCH = 8


@dataclass
class SemanticGroup:
    """A group of semantically related attributes within a class."""
    name: str
    attributes: List[str]
    description: str = ""


@dataclass
class ClassSemanticProfile:
    """Semantic analysis result for a single metamodel class."""
    class_name: str
    groups: List[SemanticGroup]
    comparable_pairs: Set[Tuple[str, str]]       # WHITELIST: pairs the LLM approved
    incomparable_pairs: Set[Tuple[str, str]]     # explicit rejections (for logging/debugging)
    analyzed: bool = True                         # True if LLM analyzed this class
    reasoning: Dict[str, str] = field(default_factory=dict)

    def is_comparable(self, attr1: str, attr2: str) -> Optional[bool]:
        """
        Whitelist approach: only pairs explicitly approved by the LLM are allowed.

        Returns:
            True  → LLM says this pair is semantically meaningful
            False → LLM analyzed this class but did NOT approve this pair
            None  → LLM did NOT analyze this class (fallback to heuristics)
        """
        # If LLM didn't analyze this class, return None → heuristics decide
        if not self.analyzed:
            return None

        key = tuple(sorted([attr1, attr2]))

        # WHITELIST: only approved pairs pass
        if key in self.comparable_pairs:
            return True

        # LLM analyzed this class but didn't approve this pair → reject
        return False


@dataclass
class SemanticCompatibilityMatrix:
    """Full semantic compatibility matrix for an entire metamodel."""
    model_name: str
    class_profiles: Dict[str, ClassSemanticProfile]
    generated_at: str = ""
    model_used: str = DEFAULT_MODEL

    def is_comparable(self, class_name: str, attr1: str, attr2: str) -> Optional[bool]:
        profile = self.class_profiles.get(class_name)
        if profile is None:
            return None
        return profile.is_comparable(attr1, attr2)

    def get_group(self, class_name: str, attr_name: str) -> Optional[str]:
        profile = self.class_profiles.get(class_name)
        if profile is None:
            return None
        for group in profile.groups:
            if attr_name in group.attributes:
                return group.name
        return "unknown"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "generated_at": self.generated_at,
            "model_used": self.model_used,
            "classes": {
                name: {
                    "groups": [
                        {"name": g.name, "attributes": g.attributes, "description": g.description}
                        for g in profile.groups
                    ],
                    "comparable_pairs": [list(p) for p in sorted(profile.comparable_pairs)],
                    "incomparable_pairs": [list(p) for p in sorted(profile.incomparable_pairs)],
                    "analyzed": profile.analyzed,
                    "reasoning": profile.reasoning,
                }
                for name, profile in self.class_profiles.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticCompatibilityMatrix":
        class_profiles = {}
        for name, cdata in data.get("classes", {}).items():
            groups = [
                SemanticGroup(name=g["name"], attributes=g["attributes"],
                              description=g.get("description", ""))
                for g in cdata.get("groups", [])
            ]
            comparable = {tuple(sorted(p)) for p in cdata.get("comparable_pairs", [])}
            incomparable = {tuple(sorted(p)) for p in cdata.get("incomparable_pairs", [])}
            analyzed = cdata.get("analyzed", True)
            reasoning = cdata.get("reasoning", {})
            class_profiles[name] = ClassSemanticProfile(
                class_name=name, groups=groups,
                comparable_pairs=comparable, incomparable_pairs=incomparable,
                analyzed=analyzed, reasoning=reasoning,
            )
        return cls(
            model_name=data.get("model_name", ""),
            class_profiles=class_profiles,
            generated_at=data.get("generated_at", ""),
            model_used=data.get("model_used", DEFAULT_MODEL),
        )


# ──────────────────────────────────────────────────────────
# Smart Batching: group classes by attribute-pair count
# ──────────────────────────────────────────────────────────

def _count_same_type_pairs(attrs: List[Dict[str, str]]) -> int:
    """Count how many same-type attribute pairs exist in a class."""
    by_type = defaultdict(list)
    for a in attrs:
        by_type[a["type"].lower()].append(a["name"])
    total = 0
    for type_attrs in by_type.values():
        n = len(type_attrs)
        total += n * (n - 1) // 2  # C(n, 2)
    return total


def _needs_llm_analysis(attrs: List[Dict[str, str]]) -> bool:
    """
    Determine if a class needs LLM analysis.
    Any class with 2+ same-type attributes has at least 1 comparison pair
    that could be semantically invalid (e.g., cpuCores > memoryMB,
    latitude > longitude). Heuristic keyword lists can't catch these
    domain-specific cases — the LLM can.
    """
    by_type = defaultdict(int)
    for a in attrs:
        by_type[a["type"].lower()] += 1
    # Need LLM if any type has 2+ attributes (at least 1 ambiguous pair)
    return any(count >= 2 for count in by_type.values())


def _create_smart_batches(classes_data: Dict[str, List[Dict[str, str]]],
                          max_pairs: int = MAX_PAIRS_PER_BATCH
                          ) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    Group classes into batches where each batch has at most max_pairs
    total same-type attribute pairs. This keeps output token count manageable.
    """
    batches = []
    current_batch = {}
    current_pairs = 0

    # Sort by pair count (smallest first) to pack efficiently
    sorted_classes = sorted(classes_data.items(),
                            key=lambda x: _count_same_type_pairs(x[1]))

    for cls_name, attrs in sorted_classes:
        pairs = _count_same_type_pairs(attrs)
        if pairs == 0:
            pairs = 1  # Minimum 1 for classes with mixed types

        if current_pairs + pairs > max_pairs and current_batch:
            batches.append(current_batch)
            current_batch = {}
            current_pairs = 0

        current_batch[cls_name] = attrs
        current_pairs += pairs

    if current_batch:
        batches.append(current_batch)

    return batches


# ──────────────────────────────────────────────────────────
# Prompt Building
# ──────────────────────────────────────────────────────────

def _build_batch_prompt(classes_data: Dict[str, List[Dict[str, str]]]) -> str:
    """Build prompt for a batch of classes.

    Prompt strategy for phi4-mini:
    - Explicit few-shot example with YES and NO cases
    - Pre-enumerated pairs with YES/NO decision per pair
    - Strict JSON output format
    """
    # Pre-enumerate all same-type pairs per class
    class_sections = []
    for cls_name, attrs in classes_data.items():
        by_type: Dict[str, List[str]] = {}
        for a in attrs:
            by_type.setdefault(a["type"], []).append(a["name"])

        pairs = []
        for type_name, names in by_type.items():
            if len(names) < 2:
                continue
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pairs.append((names[i], names[j]))

        if not pairs:
            continue

        pair_lines = "\n".join(f"  - {a} vs {b}" for a, b in pairs)
        class_sections.append(f"{cls_name}:\n{pair_lines}")

    classes_block = "\n\n".join(class_sections)

    return f"""You are a domain modeling expert for UML/OCL constraint generation. Your task: for each attribute pair, decide whether a comparison like attr1 > attr2 or attr1 = attr2 would make sense in a real-world business rule or OCL invariant.

=== CORE RULE ===
Two attributes are COMPARABLE if and only if:
1. They measure the SAME kind of real-world quantity (e.g., both are temperatures, both are dates, both are prices), AND
2. A comparison between them has a meaningful business interpretation (e.g., "startDate < endDate" enforces temporal ordering).

If EITHER condition fails, the pair is NOT comparable.

=== STEP-BY-STEP REASONING (apply for each pair) ===
Step 1: What real-world quantity does each attribute represent?
Step 2: Are they the SAME kind of quantity?
Step 3: Does comparing them make business sense?
→ If YES to all: COMPARABLE. If NO to any: NOT COMPARABLE.

=== COMPARABLE EXAMPLES (include these) ===
- startDate vs endDate → both dates in same process → "start < end" is valid → YES
- minValue vs maxValue → both are value bounds → "min <= max" is valid → YES
- minPrice vs maxPrice → both prices → "min <= max" is valid → YES
- salary vs bonus → both money amounts → "salary > bonus" is valid → YES
- departureTime vs arrivalTime → both times → "departure < arrival" is valid → YES
- currentTemp vs threshold → both temperatures → "current > threshold" triggers alert → YES
- batteryLevel vs lowBatteryThreshold → both percentages → "level < threshold" is valid → YES

=== NOT COMPARABLE EXAMPLES (exclude these) ===
- age vs salary → age is time-lived, salary is money → NO
- cpuCores vs memoryMB → count vs storage size → NO
- price vs quantity → money vs count → NO
- latitude vs longitude → different spatial axes, comparing them is meaningless → NO
- tankLevel vs mileage → fuel level vs distance → NO
- id vs price → identifier vs money → NO
- name vs date → label vs temporal → NO
- sensorId vs value → identifier vs measurement → NO
- batteryLevel vs signalStrength → different physical quantities → NO
- temperature vs humidity → different physical quantities → NO
- frequency vs power → rate vs energy → NO

=== COMMON MISTAKES TO AVOID ===
- Do NOT compare attributes just because they have the same data type (e.g., both are Integer, both are String). Type alone is not enough — they must represent the same KIND of quantity.
- Do NOT compare an identifier (id, code, serial, no, ref) with anything else.
- Do NOT compare labels/categories (name, type, kind, category, specialty, description, summary) with anything.
- Do NOT compare attributes from different physical domains (temperature vs humidity, voltage vs current, distance vs weight, dose vs frequency).
- latitude and longitude are NOT comparable — they represent different spatial axes.
- String attributes like provider, method, shift, role, status, address, phone are NOT quantities — they cannot be compared with >, <.
- When in doubt, mark as NOT comparable. False negatives are safer than false positives.

=== YOUR TASK ===
Classify each pair below. Include a pair ONLY if it is truly comparable.

{classes_block}

Respond with ONLY valid JSON, no explanation, no markdown:
{{"ClassName": [["attr1", "attr2"]], "ClassName2": []}}

Rules:
- Use [] for classes with NO comparable pairs.
- Include ALL class names from above.
- Output ONLY the JSON object, nothing else."""


# ──────────────────────────────────────────────────────────
# Streaming Ollama Call (never times out)
# ──────────────────────────────────────────────────────────

def _call_ollama_streaming(prompt: str, model: str = DEFAULT_MODEL,
                           url: str = OLLAMA_URL) -> Optional[str]:
    """
    Call Ollama API with streaming enabled.
    Reads tokens as they arrive — connection stays alive, no timeout possible.

    Uses socket-level timeout so the initial connection (which includes model
    loading into RAM) gets up to 300s, and each subsequent read also gets 300s.
    Since tokens arrive every ~0.5-2s during generation, the read timeout
    effectively never fires once generation starts.
    """
    import socket
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        response_text = []
        token_count = 0
        # 300s timeout covers model loading into RAM on first call
        with urllib.request.urlopen(req, timeout=300) as resp:
            # Set socket-level read timeout (per-read, not total)
            # This ensures we don't hang if Ollama dies mid-stream
            raw_sock = resp.fp.raw
            if hasattr(raw_sock, '_sock'):
                raw_sock._sock.settimeout(300)

            buffer = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buffer += chunk
                if chunk == b"\n":
                    line = buffer.strip()
                    buffer = b""
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            response_text.append(token)
                            token_count += 1
                            # Progress indicator every 100 tokens
                            if token_count % 100 == 0:
                                logger.info(f"    ... {token_count} tokens received")
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        full_response = "".join(response_text)
        logger.info(f"    Streamed {token_count} tokens ({len(full_response)} chars)")
        return full_response

    except urllib.error.URLError as e:
        logger.error(f"Ollama connection failed: {e}. Is 'ollama serve' running?")
        return None
    except socket.timeout as e:
        logger.error(f"Ollama read timed out: {e}. Model may be too slow or stuck.")
        return None
    except Exception as e:
        logger.error(f"Ollama streaming call failed: {e}")
        return None


# ──────────────────────────────────────────────────────────
# Response Parsing
# ──────────────────────────────────────────────────────────

def _parse_batched_response(response: str,
                            classes_data: Dict[str, List[Dict[str, str]]]
                            ) -> Dict[str, ClassSemanticProfile]:
    """Parse the batched LLM response into per-class profiles."""
    json_str = response.strip()
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse batched LLM response")
                return {}
        else:
            logger.warning("No JSON found in batched LLM response")
            return {}

    # Handle various response formats
    if isinstance(data, list):
        # LLM returned a list — try to find a dict inside or skip
        logger.warning(f"LLM returned list instead of dict, attempting recovery")
        # Try to find a dict element
        for item in data:
            if isinstance(item, dict):
                data = item
                break
        else:
            logger.warning("No dict found in list response")
            return {}

    classes_result = data.get("classes", data) if isinstance(data, dict) else {}
    if not isinstance(classes_result, dict):
        logger.warning(f"Unexpected classes_result type: {type(classes_result)}")
        return {}

    profiles = {}

    for cls_name, attrs_list in classes_data.items():
        all_attr_names = [a["name"] for a in attrs_list]
        cls_data = classes_result.get(cls_name, {})

        if not cls_data:
            profiles[cls_name] = ClassSemanticProfile(
                class_name=cls_name, groups=[],
                comparable_pairs=set(), incomparable_pairs=set()
            )
            continue

        # Handle format variations from phi4-mini:
        # Format A (expected): {"comparable": [["a","b"], ...]}
        # Format B (flat list): [["a","b"], ...] or ["a","b"]
        # Format C (dict with groups): {"groups": {...}, "comparable": [...]}
        raw_comparable = []
        groups = []

        if isinstance(cls_data, list):
            # Format B: class value is the comparable list directly
            raw_comparable = cls_data
        elif isinstance(cls_data, dict):
            # Format A or C
            raw_comparable = cls_data.get("comparable", cls_data.get("comparable_pairs", []))

            # Parse groups if present
            raw_groups = cls_data.get("groups", {})
            if isinstance(raw_groups, dict):
                for gname, gattrs in raw_groups.items():
                    if isinstance(gattrs, dict):
                        gattrs = gattrs.get("attributes", [])
                    if isinstance(gattrs, list):
                        valid = [a for a in gattrs if a in all_attr_names]
                        if valid:
                            groups.append(SemanticGroup(name=gname, attributes=valid))

        # Parse comparable pairs — handle format variations from phi4-mini:
        #   [["a","b"]]          — standard nested pairs
        #   ["a","b"]            — flat list (single pair)
        #   [["a vs b"]]         — "vs" string format
        #   [["a","b"],["c","d"]]— multiple nested pairs
        comparable = set()
        if isinstance(raw_comparable, list):
            for item in raw_comparable:
                if isinstance(item, list) and len(item) == 2:
                    a, b = item
                    if a in all_attr_names and b in all_attr_names and a != b:
                        comparable.add(tuple(sorted([a, b])))
                elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                    # ["attr1 vs attr2"] — parse "vs" format
                    parts = [p.strip() for p in item[0].split(" vs ")]
                    if len(parts) == 2:
                        a, b = parts
                        if a in all_attr_names and b in all_attr_names and a != b:
                            comparable.add(tuple(sorted([a, b])))
                elif isinstance(item, str) and " vs " in item:
                    # "attr1 vs attr2" — parse "vs" format
                    parts = [p.strip() for p in item.split(" vs ")]
                    if len(parts) == 2:
                        a, b = parts
                        if a in all_attr_names and b in all_attr_names and a != b:
                            comparable.add(tuple(sorted([a, b])))
            # Handle flat list: ["attr1", "attr2"] as a single pair
            if (len(raw_comparable) == 2
                    and all(isinstance(x, str) for x in raw_comparable)):
                a, b = raw_comparable
                if a in all_attr_names and b in all_attr_names and a != b:
                    comparable.add(tuple(sorted([a, b])))

        # Parse incomparable pairs (optional in response)
        incomparable = set()
        if isinstance(cls_data, dict):
            for pair in cls_data.get("incomparable", cls_data.get("incomparable_pairs", [])):
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = pair
                    if a in all_attr_names and b in all_attr_names and a != b:
                        incomparable.add(tuple(sorted([a, b])))

        profiles[cls_name] = ClassSemanticProfile(
            class_name=cls_name, groups=groups,
            comparable_pairs=comparable, incomparable_pairs=incomparable,
        )

    return profiles


# ──────────────────────────────────────────────────────────
# Post-Processing: Rule-based refinement of LLM results
# ──────────────────────────────────────────────────────────

# Semantic domain keywords — attributes matching these patterns belong to a domain.
# Matching is done on tokenized attribute names (camelCase / snake_case aware) to
# avoid false positives such as "updateId" being misread as containing "date".
_DOMAIN_PATTERNS = {
    "date_business": {"date", "expiry", "expires", "deadline", "birth", "due",
                      "scheduled", "booking", "departure", "arrival",
                      "checkin", "checkout"},
    "date_metadata": {"created", "updated", "modified", "timestamp", "logged",
                      "recorded", "registered", "submitted", "confirmed"},
    "distance": {"mileage", "distance", "odometer", "mile", "km",
                 "range", "depth", "altitude", "elevation"},
    "money": {"price", "cost", "amount", "salary", "wage", "fee",
              "bonus", "revenue", "budget", "balance", "payment",
              "charge", "discount", "tax", "deposit", "rent",
              "credit", "debit", "invoice", "rate", "value",
              "insurance", "premium", "worth", "fare", "total"},
    "count": {"count", "quantity", "seats", "capacity", "door",
              "cores", "threads", "slots", "items", "points", "score", "staff"},
    "geo": {"longitude", "latitude", "lon", "lng", "lat", "coordinate",
            "coord", "geo", "bearing"},
    "percentage": {"percent", "ratio", "progress",
                   "utilization", "efficiency", "coverage", "pct"},
    "temperature": {"temp", "temperature", "heat", "celsius", "fahrenheit"},
    "weight": {"weight", "mass", "kg", "ton", "pound", "lb"},
    "memory": {"memory", "ram", "cache", "storage", "byte", "bytes",
               "kb", "mb", "gb", "tb"},
    "unit": {"unit", "units", "uom", "measurement", "measure"},
    "version": {"version", "revision", "build", "release", "patch"},
    "identifier": {"id", "vin", "code", "uuid", "token", "serial", "ref",
                   "key", "identifier"},
    "label": {"name", "title", "label", "description", "category",
              "status", "kind", "tag"},
    "time": {"time", "hour", "minute", "duration", "day", "week"},
    "level": {"level", "tank", "fuel"},
    "power": {"horsepower", "hp", "watt", "kilowatt", "torque"},
    "age": {"age"},
    "model_year": {"year"},
}

_DOMAIN_FAMILY = {
    "date_business": "temporal_business",
    "date_metadata": "temporal_metadata",
    "time": "temporal_business",
    "distance": "distance",
    "money": "money",
    "count": "count",
    "percentage": "percentage",
    "temperature": "temperature",
    "weight": "weight",
    "memory": "memory",
    "geo": "geo",
    "unit": "unit",
    "version": "version",
    "identifier": "identifier",
    "label": "label",
    "level": "level",
    "power": "power",
    "age": "age",
    "model_year": "model_year",
}

# Only these domain families are safe to auto-approve when the LLM missed them.
# Label-like domains remain conservative: they can remove cross-domain mistakes,
# but we do not auto-add them solely based on a shared coarse category.
_AUTO_APPROVE_FAMILIES = {
    "temporal_business",
    "temporal_metadata",
    "distance",
    "money",
    "count",
    "percentage",
    "temperature",
    "weight",
    "memory",
    "level",
}

# Domains that are NEVER comparable with anything outside their own family.
# If either attribute in a pair belongs to one of these, and the other does NOT
# belong to the same family, the pair is always rejected.
_NEVER_CROSS_FAMILIES = {
    "identifier",   # id, code, serial, etc. — never compare with anything
    "label",        # name, title, description — never compare with anything
    "unit",         # unit names — never compare
    "version",      # version numbers — never compare with non-versions
    "geo",          # longitude/latitude — only comparable with other coordinates
}

# Additional keyword patterns for attributes that should never be in comparisons.
# These catch cases like "specialty", "shift", "method", "type", "summary" etc.
# that don't match any domain but are clearly not measurable quantities.
_NON_MEASURABLE_KEYWORDS = {
    "type", "kind", "category", "class", "mode", "method", "format",
    "specialty", "shift", "role", "status", "state", "phase",
    "summary", "note", "comment", "remark", "reason",
    "address", "email", "phone", "url", "path",
    "provider", "vendor", "supplier", "manufacturer",
    "gender", "sex", "nationality", "ethnicity",
    "color", "colour", "shape", "size",
    "no",  # as in "roomNo", "policyNo" — identifiers
}

# Compound rules: if attr contains BOTH a qualifier and a domain keyword, classify by domain keyword
# e.g., "mileageStart" → contains "mileage" (distance) + "start" → distance, not date
# e.g., "dateFrom" → contains "date" (date_business) + "from" → date_business
# The qualifier alone ("start", "end", "from", "to") is NOT enough to classify.
_QUALIFIER_WORDS = {"start", "end", "from", "to", "min", "max", "begin", "finish"}


def _tokenize_attr_name(attr_name: str) -> List[str]:
    """Split an attribute name into lowercase semantic tokens.

    Handles camelCase, PascalCase, snake_case, and simple acronym boundaries.
    Examples:
        "updateId" -> ["update", "id"]
        "memoryMB" -> ["memory", "mb"]
        "scheduled_at" -> ["scheduled", "at"]
    """
    import re

    normalized = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', attr_name)
    normalized = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', normalized)
    raw_tokens = [t.lower() for t in re.split(r'[^A-Za-z0-9]+', normalized) if t]

    expanded = set(raw_tokens)
    for token in list(expanded):
        if token.endswith('s') and len(token) > 3:
            expanded.add(token[:-1])

    return sorted(expanded)


def _classify_attr_domain(attr_name: str) -> Optional[str]:
    """Classify an attribute into a semantic domain by keyword matching.

    Uses robust tokenization and prioritizes domain keywords over qualifiers.
    e.g., "mileageStart" → split to {mileage, start} → "mileage" matches distance → distance
    e.g., "startDate" → split to {start, date} → "date" matches date_business → date_business
    """
    lower = attr_name.lower()
    tokens = set(_tokenize_attr_name(attr_name))

    # Remove qualifier words from domain matching — they don't indicate domain
    domain_parts = tokens - _QUALIFIER_WORDS

    # Match domain keywords against normalized tokens first.
    scores = {}
    for domain, keywords in _DOMAIN_PATTERNS.items():
        for kw in keywords:
            if kw in domain_parts:
                scores[domain] = scores.get(domain, 0) + 2  # strong match
            # Conservative fallback for compact forms like "timestamped".
            elif any(p.startswith(kw) or p.endswith(kw) for p in domain_parts):
                scores[domain] = scores.get(domain, 0) + 1  # weaker match

    if not scores:
        return None
    return max(scores, key=scores.get)


def _extract_root_and_qualifier(attr_name: str) -> Tuple[str, Optional[str]]:
    """Extract root word and qualifier from camelCase attribute names.

    Examples:
        "mileageStart" → ("mileage", "start")
        "dateFrom"     → ("date", "from")
        "startDate"    → ("date", "start")
        "price"        → ("price", None)
    """
    parts = _tokenize_attr_name(attr_name)
    if not parts:
        return (attr_name.lower(), None)

    qualifiers_found = []
    root_parts = []
    for p in parts:
        if p in _QUALIFIER_WORDS:
            qualifiers_found.append(p)
        else:
            root_parts.append(p)

    root = "".join(root_parts) if root_parts else attr_name.lower()
    qualifier = qualifiers_found[0] if qualifiers_found else None
    return (root, qualifier)


def _domain_family(domain: Optional[str]) -> Optional[str]:
    if domain is None:
        return None
    return _DOMAIN_FAMILY.get(domain, domain)


def _is_non_measurable(attr_name: str) -> bool:
    """Check if an attribute is clearly non-measurable (label, category, etc.)."""
    tokens = set(_tokenize_attr_name(attr_name))
    return bool(tokens & _NON_MEASURABLE_KEYWORDS)


def _postprocess_profiles(profiles: Dict[str, ClassSemanticProfile],
                          classes_data: Dict[str, List[Dict[str, str]]]
                          ) -> Dict[str, ClassSemanticProfile]:
    """
    Rule-based post-processing to fix common LLM mistakes:
    1. Same-root pairs (mileageStart/mileageEnd, dateFrom/dateTo) → always comparable
    2. Same-domain pairs → add if missing
    3. Different-domain pairs → remove if wrongly added
    """
    for cls_name, profile in profiles.items():
        attrs = classes_data.get(cls_name, [])
        if not attrs:
            continue

        all_attr_names = [a["name"] for a in attrs]

        # Classify each attribute
        domains = {}
        roots = {}
        for a in attrs:
            domain = _classify_attr_domain(a["name"])
            if domain:
                domains[a["name"]] = domain
            root, qualifier = _extract_root_and_qualifier(a["name"])
            roots[a["name"]] = (root, qualifier)

        # Group by type for same-type pairs only
        by_type: Dict[str, List[str]] = {}
        for a in attrs:
            by_type.setdefault(a["type"], []).append(a["name"])

        added = set()
        removed = set()

        for type_name, type_attrs in by_type.items():
            if len(type_attrs) < 2:
                continue

            for i in range(len(type_attrs)):
                for j in range(i + 1, len(type_attrs)):
                    a, b = type_attrs[i], type_attrs[j]
                    key = tuple(sorted([a, b]))
                    da = domains.get(a)
                    db = domains.get(b)
                    fa = _domain_family(da)
                    fb = _domain_family(db)
                    root_a, qual_a = roots.get(a, (a, None))
                    root_b, qual_b = roots.get(b, (b, None))

                    # Rule 0: Non-measurable attributes → NEVER comparable
                    # e.g., specialty, type, method, summary, provider, shift
                    if _is_non_measurable(a) or _is_non_measurable(b):
                        if key in profile.comparable_pairs:
                            profile.comparable_pairs.discard(key)
                            profile.incomparable_pairs.add(key)
                            removed.add(key)
                        continue

                    # Rule 0b: "Never-cross" domains → reject if either is identifier/label
                    # e.g., doctorId vs anything, name vs anything
                    if fa in _NEVER_CROSS_FAMILIES or fb in _NEVER_CROSS_FAMILIES:
                        # Only allow if BOTH are in the SAME never-cross family
                        if fa != fb:
                            if key in profile.comparable_pairs:
                                profile.comparable_pairs.discard(key)
                                profile.incomparable_pairs.add(key)
                                removed.add(key)
                            continue

                    # Rule 1: Same root + different qualifiers → ALWAYS comparable
                    # e.g., mileageStart/mileageEnd, dateFrom/dateTo, minPrice/maxPrice
                    if root_a == root_b and qual_a and qual_b and qual_a != qual_b:
                        if key not in profile.comparable_pairs:
                            profile.comparable_pairs.add(key)
                            added.add(key)
                        continue  # skip domain check — root match is definitive

                    # Rule 2: Same compatible family → comparable for strong domains only
                    if fa and fb:
                        if fa == fb:
                            if fa in _AUTO_APPROVE_FAMILIES and key not in profile.comparable_pairs:
                                profile.comparable_pairs.add(key)
                                added.add(key)
                        else:
                            # Different semantic families → NOT comparable
                            if key in profile.comparable_pairs:
                                profile.comparable_pairs.discard(key)
                                profile.incomparable_pairs.add(key)
                                removed.add(key)

        if added or removed:
            logger.info(f"  Post-process {cls_name}: +{len(added)} -{len(removed)} "
                        f"(added: {added}, removed: {removed})")

    return profiles


# ──────────────────────────────────────────────────────────
# Main Analyzer
# ──────────────────────────────────────────────────────────

class LLMSemanticAnalyzer:
    """
    Analyzes metamodel attributes using Phi-4 via Ollama to build
    a semantic compatibility matrix.

    Strategy:
      1. Filter: skip classes where heuristics suffice (<3 same-type attrs)
      2. Smart batch: group remaining classes by pair count (~30 pairs/batch)
      3. Stream: call Ollama with stream=True (never times out)
      4. Cache: save result to JSON, instant on subsequent runs

    Usage:
        analyzer = LLMSemanticAnalyzer()
        matrix = analyzer.analyze_metamodel(metamodel)
    """

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL,
                 cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.model = model
        self.ollama_url = ollama_url
        self.cache_dir = cache_dir or CACHE_DIR
        self.use_cache = use_cache

    def _cache_key(self, metamodel) -> str:
        parts = []
        if isinstance(metamodel.classes, dict):
            class_names = sorted(metamodel.classes.keys())
        else:
            class_names = sorted(c.name for c in metamodel.classes)
        for cls_name in class_names:
            attrs = sorted(
                [(a.name, a.type) for a in metamodel.get_attributes_for(cls_name)],
                key=lambda x: x[0]
            )
            parts.append(f"{cls_name}:{attrs}")
        content = "|".join(parts)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _load_cache(self, cache_key: str, model_name: str) -> Optional[SemanticCompatibilityMatrix]:
        if not self.use_cache:
            return None
        cache_file = self.cache_dir / f"{model_name}_{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded semantic cache from {cache_file}")
                return SemanticCompatibilityMatrix.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_cache(self, matrix: SemanticCompatibilityMatrix, cache_key: str, model_name: str):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{model_name}_{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(matrix.to_dict(), f, indent=2)
            logger.info(f"Saved semantic cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _check_ollama_available(self) -> bool:
        import urllib.request
        import urllib.error
        try:
            base_url = self.ollama_url.rsplit("/api/", 1)[0]
            tags_url = f"{base_url}/api/tags"
            req = urllib.request.Request(
                tags_url,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                available = any(
                    self.model in m or f"{self.model}:latest" in m
                    for m in models
                )
                if not available:
                    logger.warning(
                        f"Model '{self.model}' not found in Ollama. "
                        f"Available: {models}. Run: ollama pull {self.model}"
                    )
                return available
        except Exception as e:
            logger.warning(f"Ollama not reachable at {self.ollama_url}: {e}")
            return False

    def analyze_metamodel(self, metamodel, model_name: str = "unknown") -> SemanticCompatibilityMatrix:
        """
        Analyze an entire metamodel using smart batching + streaming.

        Flow:
          1. Check cache → return instantly if found
          2. Filter classes: skip those with <3 same-type attrs
          3. Smart batch by pair count
          4. Stream each batch to Phi-4
          5. Merge results, cache, and return
        """
        cache_key = self._cache_key(metamodel)

        # Try cache first
        cached = self._load_cache(cache_key, model_name)
        if cached is not None:
            return cached

        # Check Ollama availability
        if not self._check_ollama_available():
            logger.warning(
                "Ollama not available — falling back to empty semantic matrix. "
                "Heuristic rules will be used instead."
            )
            return SemanticCompatibilityMatrix(
                model_name=model_name, class_profiles={},
                generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                model_used=self.model,
            )

        # Collect all classes with attributes
        if isinstance(metamodel.classes, dict):
            class_items = list(metamodel.classes.items())
        else:
            class_items = [(cls.name, cls) for cls in metamodel.classes]

        all_classes_data = {}
        llm_classes_data = {}
        skipped_classes = []

        for cls_name, _ in class_items:
            attrs = metamodel.get_attributes_for(cls_name)
            if len(attrs) < 2:
                continue
            attr_list = [{"name": a.name, "type": a.type} for a in attrs]
            all_classes_data[cls_name] = attr_list

            if _needs_llm_analysis(attr_list):
                llm_classes_data[cls_name] = attr_list
            else:
                skipped_classes.append(cls_name)

        logger.info(f"{'='*60}")
        logger.info(f"LLM Semantic Analysis: {model_name}")
        logger.info(f"Model: {self.model}")
        logger.info(f"  Total classes: {len(class_items)}")
        logger.info(f"  Classes with 2+ attrs: {len(all_classes_data)}")
        logger.info(f"  Need LLM (3+ same-type attrs): {len(llm_classes_data)}")
        logger.info(f"  Skipped (heuristics sufficient): {len(skipped_classes)}")
        if skipped_classes:
            logger.info(f"    → {', '.join(skipped_classes)}")
        logger.info(f"{'='*60}")

        class_profiles = {}
        total_start = time.time()

        if llm_classes_data:
            # Smart batch by attribute-pair count
            batches = _create_smart_batches(llm_classes_data, MAX_PAIRS_PER_BATCH)
            logger.info(f"Created {len(batches)} smart batches (max ~{MAX_PAIRS_PER_BATCH} pairs each)")

            for i, batch in enumerate(batches, 1):
                batch_pairs = sum(_count_same_type_pairs(attrs) for attrs in batch.values())
                batch_classes = list(batch.keys())
                logger.info(f"\n  Batch {i}/{len(batches)}: {len(batch)} classes, ~{batch_pairs} pairs")
                logger.info(f"    Classes: {', '.join(batch_classes)}")

                prompt = _build_batch_prompt(batch)
                start = time.time()
                response = _call_ollama_streaming(prompt, model=self.model, url=self.ollama_url)
                elapsed = time.time() - start
                logger.info(f"    → {self.model} responded in {elapsed:.1f}s")

                if response:
                    batch_profiles = _parse_batched_response(response, batch)
                    batch_profiles = _postprocess_profiles(batch_profiles, batch)
                    class_profiles.update(batch_profiles)

                    for cn, prof in batch_profiles.items():
                        logger.info(f"    {cn}: {len(prof.groups)} groups, "
                                    f"{len(prof.comparable_pairs)} comp, "
                                    f"{len(prof.incomparable_pairs)} incomp")
                else:
                    logger.warning(f"    Batch {i} failed — classes will use heuristic fallback")

        # Add empty profiles for skipped / failed classes
        # Mark as analyzed=False so they fall back to heuristics
        for cls_name, _ in class_items:
            if cls_name not in class_profiles:
                class_profiles[cls_name] = ClassSemanticProfile(
                    class_name=cls_name, groups=[],
                    comparable_pairs=set(), incomparable_pairs=set(),
                    analyzed=False,  # NOT analyzed → heuristics decide
                )

        total_elapsed = time.time() - total_start
        total_comp = sum(len(p.comparable_pairs) for p in class_profiles.values())
        total_incomp = sum(len(p.incomparable_pairs) for p in class_profiles.values())
        total_groups = sum(len(p.groups) for p in class_profiles.values())

        logger.info(f"\n{'='*60}")
        logger.info(f"Semantic analysis complete in {total_elapsed:.1f}s")
        logger.info(f"  {total_groups} semantic groups across {len(class_profiles)} classes")
        logger.info(f"  {total_comp} comparable pairs, {total_incomp} incomparable pairs")
        logger.info(f"{'='*60}")

        matrix = SemanticCompatibilityMatrix(
            model_name=model_name,
            class_profiles=class_profiles,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            model_used=self.model,
        )

        self._save_cache(matrix, cache_key, model_name)
        return matrix
