"""Unit tests for the OCL complexity metrics in
``modules.generation.benchmark.complexity_calculator``.

They pin two measurement rules:

1. WNO/OC tokenise operators so each occurrence is consumed exactly once
   ('<=' is one comparison, not '<' + '=' + '<='; the '>' of '->' is not a
   comparison).
2. NNR-C / DN-CA (and the other chain metrics) count navigation steps only:
   a method call such as ``.abs()`` or ``.size()`` is not a step, and an
   iterator variable is resolved back to the collection it ranges over.

Run from the repository root with ``python3 -m pytest tests/ -q``.
"""
import logging
from pathlib import Path

import pytest

from modules.generation.benchmark.complexity_calculator import (
    ComplexityWeights,
    compute_cic,
    compute_dn_ca,
    compute_nnc_plus,
    compute_nnr_c,
    compute_oc,
    compute_wnc,
    compute_wno,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Table 1 of the paper (Jha, Monahan, Wu - STAF 2025): the operator weights the
# paper benchmark suite (examples/paper_ten_models_vgcr.yaml) supplies. The
# module's DEFAULT_OPERATOR_WEIGHTS additionally weight arithmetic/string
# helpers (abs, mod, concat, ...) at 1.0 so that WNO is a strict superset of
# OC; the "abs" case below is therefore checked under both tables.
TABLE1 = ComplexityWeights(operator_weights={
    '=': 1.0, '<>': 1.0, '>': 1.0, '<': 1.0, '>=': 1.0, '<=': 1.0,
    'and': 1.5, 'or': 1.5, 'not': 1.5,
    'implies': 2.0, 'xor': 2.0,
    'select': 2.0, 'collect': 2.0, 'reject': 2.0, 'flatten': 2.0,
    'forAll': 3.0, 'exists': 3.0, 'closure': 3.0, 'iterate': 3.0,
    'oclAsType': 2.0, 'asSet': 1.5,
})

FORALL = "(self.rentals->forAll(x | x.lateFee > 2)) and (self.vin.size() <= 20)"


# ---------------------------------------------------------------------------
# WNO: each operator occurrence is consumed exactly once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ocl, expected", [
    ("self.branch <> null", 1.0),                       # '<>' is one operator
    ("self.amount.abs() <= 9", 1.0),                    # '<=' is one operator
    ("self.company.isActive", 0.0),
    ("self.mileage > self.year", 1.0),
    ("self.discount <= self.amount and self.amount <= self.tax", 3.5),
    (FORALL, 6.5),                                      # forAll 3 + '>' 1 + and 1.5 + '<=' 1
])
def test_wno_counts_each_operator_once(ocl, expected):
    assert compute_wno(ocl, TABLE1) == expected


def test_wno_arrow_is_not_a_comparison():
    # The '>' of the collection arrow must not be counted.
    assert compute_wno("self.rentals->size() = 2", TABLE1) == 1.0
    assert compute_wno("self.rentals->notEmpty()", TABLE1) == 0.0


def test_wno_default_weights_count_abs_once():
    # Under the module defaults 'abs' carries weight 1.0 and, like '<=', is
    # counted exactly once: 1.0 (abs) + 1.0 (<=).
    assert compute_wno("self.amount.abs() <= 9") == 2.0
    assert compute_wno("self.branch <> null") == 1.0


def test_oc_uses_the_same_tokenisation():
    assert compute_oc("self.branch <> null") == 1.0
    assert compute_oc("self.amount.abs() <= 9") == 1.0
    assert compute_oc("self.discount <= self.amount and self.amount <= self.tax") == 3.5


# ---------------------------------------------------------------------------
# NNR-C / DN-CA: method calls are not navigation steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ocl, nnr_c, dn_ca", [
    ("self.branch <> null", 1.0, 1),
    ("self.amount.abs() <= 9", 1.0, 1),                 # .abs() is a call, not a step
    ("self.company.isActive", 2.0, 2),                  # trailing attribute still counts
    ("self.mileage > self.year", 2.0, 1),
    ("self.discount <= self.amount and self.amount <= self.tax", 4.0, 1),
    ("self.vin.size() <= 20", 1.0, 1),
])
def test_navigation_metrics_without_metamodel(ocl, nnr_c, dn_ca):
    assert compute_nnr_c(ocl) == nnr_c
    assert compute_dn_ca(ocl) == dn_ca


def test_iterator_variable_resolves_to_its_collection():
    # x ranges over self.rentals, so x.lateFee is the path self.rentals.lateFee
    # (length 2) inside the iterator body: 2 * 1.5 = 3.0.
    assert compute_dn_ca(FORALL) == 3.0
    # Without a metamodel every reference weighs 1: rentals + lateFee + vin.
    assert compute_nnr_c(FORALL) == 3.0
    assert compute_cic(FORALL) == 3.0
    # The iterator variable itself is not a navigation step in the derived
    # chain metrics either.
    assert compute_nnc_plus(FORALL) == 4.0      # rentals (1) + rentals.lateFee (2) + vin (1)
    assert compute_wnc(FORALL) == 6.0           # 1 + 2*2 (inside forAll) + 1


def test_forall_with_metamodel_weights_collection_end():
    from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor

    logging.disable(logging.CRITICAL)
    try:
        mm = MetamodelExtractor(str(REPO_ROOT / 'models' / 'car_rental.xmi')).metamodel
    finally:
        logging.disable(logging.NOTSET)

    # rentals is a 1..* end of Vehicle (weight 2); lateFee is an attribute of
    # Rental, found because x is resolved to Rental; vin is a Vehicle attribute.
    assert compute_nnr_c(FORALL, mm, 'Vehicle') == 4.0
    assert compute_dn_ca(FORALL) == 3.0
    assert compute_wno(FORALL, TABLE1) == 6.5
    assert compute_cic(FORALL) == 3.0
    # Metamodel-aware values for the simple cases are unchanged.
    assert compute_nnr_c("self.mileage > self.year", mm, 'Vehicle') == 2.0
    assert compute_nnr_c("self.vin.size() <= 20", mm, 'Vehicle') == 1.0
