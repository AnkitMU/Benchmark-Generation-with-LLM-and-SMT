#!/usr/bin/env python3
"""Semantics-preserving OCL invariant encoder for the bounded Z3 model used by
``GenericGlobalConsistencyChecker``.

The pattern-specific encoders in ``generic_global_consistency_checker.py``
recognise fragments of an invariant with regular expressions.  Whatever they do
not recognise is silently dropped, which under-approximates the constraint and
makes a joint "consistent" verdict unsound (the solver is asked a weaker
question than the one the OCL text states).

This module takes the opposite approach:

* the whole invariant is parsed into an AST (``OCLParser``);
* every node is evaluated against the shared variables created by the checker
  (presence bits, attribute variables, relation matrices and functional
  reference variables), with iterator variables bound to concrete object slots
  (``GenericOCLEncoder``);
* any construct that cannot be encoded faithfully raises ``UnsupportedOCL``.
  Nothing is ever dropped.

Semantics follow the bounded-model conventions already used by the checker:

* quantification ranges over *present* object slots only;
* a navigation through an absent optional reference is *invalid*; any
  comparison involving an invalid value evaluates to ``false`` (so an invariant
  that depends on it is violated, as in USE);
* attribute values are never ``null`` (``self.attr <> null`` is ``true``);
* ``EString`` attributes whose name looks like a date are Int-coded by the
  checker's ``DateAdapter``; string literals compared against them are mapped
  with the checker's ``_string_to_int`` (the existing convention).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from z3 import (And, AtLeast, AtMost, BoolVal, Concat, Contains, If, Implies,
                IndexOf, InRe, IntVal, Length, Not, Or, PbEq, PrefixOf, RealVal,
                StringSort, StringVal, SubString, SuffixOf, Sum, ToInt, ToReal,
                Xor, is_bool, is_int, is_real, is_true)


class UnsupportedOCL(Exception):
    """Raised when an invariant contains a construct this encoder cannot encode
    faithfully.  The caller must not treat the invariant as encoded."""


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer / parser
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORDS = {
    'and', 'or', 'not', 'implies', 'xor', 'if', 'then', 'else', 'endif',
    'let', 'in', 'true', 'false', 'null', 'self', 'div', 'mod',
}

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>\d+(?:\.\d+)?)
  | (?P<str>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<ident>[A-Za-z_]\w*(?:::\w+)*)
  | (?P<op>->|<>|<=|>=|==|!=|::|[.()\[\]{},|=<>+\-*/#@:])
""", re.X)

_REL_OPS = {'=', '<>', '<', '>', '<=', '>=', '==', '!='}


def tokenize(text: str) -> List[Tuple[str, str]]:
    toks: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise UnsupportedOCL(f"unexpected character {text[pos]!r} at {pos}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind == 'ws':
            continue
        if kind == 'ident' and val in _KEYWORDS:
            kind = 'kw'
        elif kind == 'str':
            val = val[1:-1]
        elif kind == 'op':
            val = {'==': '=', '!=': '<>'}.get(val, val)
        toks.append((kind, val))
    toks.append(('eof', ''))
    return toks


class OCLParser:
    """Recursive-descent parser for the OCL expression sublanguage.

    Precedence (low → high): let, implies, xor, or, and, not, relational,
    additive, multiplicative, unary minus, postfix (``.``, ``->``).
    """

    def __init__(self, text: str):
        self.toks = tokenize(text)
        self.pos = 0

    # -- token helpers ------------------------------------------------------
    def peek(self, k: int = 0) -> Tuple[str, str]:
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def next(self) -> Tuple[str, str]:
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def at(self, kind: str, val: Optional[str] = None, k: int = 0) -> bool:
        t = self.peek(k)
        return t[0] == kind and (val is None or t[1] == val)

    def expect(self, kind: str, val: Optional[str] = None) -> Tuple[str, str]:
        if not self.at(kind, val):
            got = self.peek()
            raise UnsupportedOCL(
                f"expected {val or kind}, got {got[1]!r} ({got[0]}) at token {self.pos}")
        return self.next()

    # -- grammar ------------------------------------------------------------
    def parse(self) -> Any:
        node = self.parse_expr()
        if not self.at('eof'):
            got = self.peek()
            raise UnsupportedOCL(f"trailing input starting at {got[1]!r}")
        return node

    def parse_expr(self) -> Any:
        if self.at('kw', 'let'):
            return self.parse_let()
        return self.parse_implies()

    def parse_let(self) -> Any:
        self.expect('kw', 'let')
        name = self.expect('ident')[1]
        if self.at('op', ':'):
            self.next()
            self.expect('ident')  # type annotation (ignored)
        self.expect('op', '=')
        value = self.parse_expr()
        self.expect('kw', 'in')
        body = self.parse_expr()
        return ('let', name, value, body)

    def _parse_left_assoc(self, sub, kws: Tuple[str, ...]) -> Any:
        left = sub()
        while self.peek()[0] == 'kw' and self.peek()[1] in kws:
            op = self.next()[1]
            right = sub()
            left = ('bin', op, left, right)
        return left

    def parse_implies(self) -> Any:
        return self._parse_left_assoc(self.parse_xor, ('implies',))

    def parse_xor(self) -> Any:
        return self._parse_left_assoc(self.parse_or, ('xor',))

    def parse_or(self) -> Any:
        return self._parse_left_assoc(self.parse_and, ('or',))

    def parse_and(self) -> Any:
        return self._parse_left_assoc(self.parse_not, ('and',))

    def parse_not(self) -> Any:
        if self.at('kw', 'not'):
            self.next()
            return ('un', 'not', self.parse_not())
        return self.parse_rel()

    def parse_rel(self) -> Any:
        left = self.parse_add()
        if self.peek()[0] == 'op' and self.peek()[1] in _REL_OPS:
            op = self.next()[1]
            right = self.parse_add()
            if self.peek()[0] == 'op' and self.peek()[1] in _REL_OPS:
                raise UnsupportedOCL("chained relational operators")
            return ('bin', op, left, right)
        return left

    def parse_add(self) -> Any:
        left = self.parse_mul()
        while self.peek()[0] == 'op' and self.peek()[1] in ('+', '-'):
            op = self.next()[1]
            right = self.parse_mul()
            left = ('bin', op, left, right)
        return left

    def parse_mul(self) -> Any:
        left = self.parse_unary()
        while ((self.peek()[0] == 'op' and self.peek()[1] in ('*', '/'))
               or (self.peek()[0] == 'kw' and self.peek()[1] in ('div', 'mod'))):
            op = self.next()[1]
            right = self.parse_unary()
            left = ('bin', op, left, right)
        return left

    def parse_unary(self) -> Any:
        if self.at('op', '-'):
            self.next()
            return ('un', '-', self.parse_unary())
        if self.at('kw', 'not'):
            self.next()
            return ('un', 'not', self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self) -> Any:
        node = self.parse_primary()
        while True:
            if self.at('op', '.'):
                self.next()
                name = self.expect('ident')[1]
                if self.at('op', '('):
                    args = self.parse_args()
                    node = ('call', node, name, args)
                else:
                    node = ('nav', node, name)
            elif self.at('op', '->'):
                self.next()
                name = self.expect('ident')[1]
                if not self.at('op', '('):
                    node = ('arrow', node, name, [], [], None)
                    continue
                self.expect('op', '(')
                iters = self._try_parse_iterators()
                if iters is not None:
                    body = self.parse_expr()
                    self.expect('op', ')')
                    node = ('arrow', node, name, [], iters, body)
                else:
                    args = []
                    if not self.at('op', ')'):
                        args.append(self.parse_expr())
                        while self.at('op', ','):
                            self.next()
                            args.append(self.parse_expr())
                    self.expect('op', ')')
                    node = ('arrow', node, name, args, [], None)
            elif self.at('op', '@'):
                raise UnsupportedOCL("@pre is not supported in invariants")
            else:
                return node

    def _try_parse_iterators(self) -> Optional[List[str]]:
        """If the upcoming tokens look like ``v1 [: T] (, v2 [: T])* |`` consume
        them and return the iterator names; otherwise leave the position
        unchanged and return None."""
        save = self.pos
        names: List[str] = []
        while True:
            if not self.at('ident'):
                self.pos = save
                return None
            names.append(self.next()[1])
            if self.at('op', ':'):
                self.next()
                if not self.at('ident'):
                    self.pos = save
                    return None
                self.next()
            if self.at('op', ','):
                self.next()
                continue
            if self.at('op', '|'):
                self.next()
                return names
            self.pos = save
            return None

    def parse_args(self) -> List[Any]:
        self.expect('op', '(')
        args: List[Any] = []
        if not self.at('op', ')'):
            args.append(self.parse_expr())
            while self.at('op', ','):
                self.next()
                args.append(self.parse_expr())
        self.expect('op', ')')
        return args

    def parse_primary(self) -> Any:
        kind, val = self.peek()
        if kind == 'num':
            self.next()
            return ('num', float(val) if '.' in val else int(val))
        if kind == 'str':
            self.next()
            return ('str', val)
        if kind == 'kw':
            if val in ('true', 'false'):
                self.next()
                return ('bool', val == 'true')
            if val == 'null':
                self.next()
                return ('null',)
            if val == 'self':
                self.next()
                return ('var', 'self')
            if val == 'if':
                return self.parse_if()
            raise UnsupportedOCL(f"unexpected keyword {val!r}")
        if kind == 'op':
            if val == '(':
                self.next()
                node = self.parse_expr()
                self.expect('op', ')')
                return node
            if val == '#':
                self.next()
                return ('enum', self.expect('ident')[1])
            raise UnsupportedOCL(f"unexpected token {val!r}")
        if kind == 'ident':
            self.next()
            if val in ('Set', 'Bag', 'Sequence', 'OrderedSet') and self.at('op', '{'):
                self.next()
                items: List[Any] = []
                if not self.at('op', '}'):
                    items.append(self.parse_expr())
                    while self.at('op', ','):
                        self.next()
                        items.append(self.parse_expr())
                self.expect('op', '}')
                return ('collit', items)
            if val == 'Tuple' and self.at('op', '{'):
                raise UnsupportedOCL("tuple literals")
            if '::' in val:
                return ('enum', val.split('::')[-1])
            return ('var', val)
        raise UnsupportedOCL("unexpected end of expression")

    def parse_if(self) -> Any:
        self.expect('kw', 'if')
        cond = self.parse_expr()
        self.expect('kw', 'then')
        then = self.parse_expr()
        self.expect('kw', 'else')
        els = self.parse_expr()
        self.expect('kw', 'endif')
        return ('if', cond, then, els)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime values
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scalar:
    """A single primitive value.  ``defined`` is a Z3 Bool that is true when the
    value is well-defined (navigation did not pass through an absent optional
    reference, divisor non-zero, ...)."""
    expr: Any
    kind: str                         # 'int' | 'real' | 'bool' | 'string'
    defined: Any = None               # None means trivially defined
    literal: Any = None               # raw Python literal if this is a literal
    case: Optional[str] = None        # 'upper' / 'lower' after toUpperCase()/toLowerCase()
    ecore_string: bool = False        # Int-coded by DateAdapter but declared EString
    indexof: Any = None               # (haystack Scalar, needle expr) for an indexOf() result
    card: Any = None                  # list of guards when this is a collection ->size():
                                      # comparisons with a literal become pseudo-Boolean
                                      # cardinality constraints (AtLeast/AtMost), which the
                                      # SAT core handles far better than Sum(If(...)) in LIA.
    alts: Any = None                  # [(guard, expr)] when the value comes from a navigation:
                                      # ``expr`` is the If-chain over them; comparisons and
                                      # string operations are distributed over the alternatives
                                      # instead, which keeps Z3's sequence solver away from
                                      # If-terms (mutually exclusive guards => equivalent).


@dataclass
class ObjSet:
    """A (possibly singleton) collection of objects.  Each element is
    ``(guard, class_name, slot_index)``; the element belongs to the collection
    exactly when ``guard`` holds."""
    elems: List[Tuple[Any, str, int]]
    single: bool = False              # statically at most one element


@dataclass
class ValSet:
    """A collection of primitive values: ``(guard, Scalar)`` pairs."""
    elems: List[Tuple[Any, Scalar]]


@dataclass
class TypeRef:
    name: str


class NullVal:
    pass


Value = Union[Scalar, ObjSet, ValSet, TypeRef, NullVal]

_TRUE = BoolVal(True)
_FALSE = BoolVal(False)


def _is_trivially_true(e) -> bool:
    return e is None or (is_bool(e) and is_true(e))


def _and(*parts):
    parts = [p for p in parts if not _is_trivially_true(p)]
    if not parts:
        return _TRUE
    if len(parts) == 1:
        return parts[0]
    return And(*parts)


def _or(parts: List[Any]):
    parts = list(parts)
    if not parts:
        return _FALSE
    if len(parts) == 1:
        return parts[0]
    return Or(*parts)


def _kind_of(var) -> str:
    if is_bool(var):
        return 'bool'
    if is_int(var):
        return 'int'
    if is_real(var):
        return 'real'
    try:
        if var.sort() == StringSort():
            return 'string'
    except Exception:
        pass
    return 'int'


def _default_for(kind: str):
    return {'bool': _FALSE, 'int': IntVal(0), 'real': RealVal(0),
            'string': StringVal('')}.get(kind, IntVal(0))


def _ite_chain(alts: List[Tuple[Any, Any]], kind: str):
    """Build ``If(g0, e0, If(g1, e1, ... e_last))`` from guarded alternatives."""
    if not alts:
        return _default_for(kind)
    acc = alts[-1][1]
    for g, e in reversed(alts[:-1]):
        acc = If(g, e, acc)
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

_INV_HEADER_RE = re.compile(r'\binv\s*\w*\s*:', re.IGNORECASE)


class GenericOCLEncoder:
    """Encode whole OCL invariants against a checker's shared variables.

    ``checker`` is a ``GenericGlobalConsistencyChecker``; this class only uses
    its ``extractor``, ``classes``, ``_get_z3_type``, ``_string_to_int``,
    ``_ocl_regex_to_z3`` and ``_case_variant_regex_for_literal`` members.
    """

    NUMERIC = ('int', 'real')

    def __init__(self, checker):
        self.checker = checker
        self.extractor = checker.extractor
        self._ast_cache: Dict[str, Any] = {}

    # ── public API ────────────────────────────────────────────────────────

    @staticmethod
    def strip_header(text: str) -> List[str]:
        """Return the invariant bodies of an OCL snippet (``context`` line and
        ``inv [name]:`` prefixes removed).  A snippet without ``inv:`` is taken
        to be a bare expression."""
        text = text.strip()
        if 'pre:' in text or 'post:' in text or 'body:' in text or 'def:' in text:
            raise UnsupportedOCL("only invariants are supported")
        parts = _INV_HEADER_RE.split(text)
        if len(parts) == 1:
            if re.match(r'^\s*context\b', text, re.IGNORECASE):
                raise UnsupportedOCL("context without inv:")
            return [text]
        bodies = [p.strip() for p in parts[1:] if p.strip()]
        if not bodies:
            raise UnsupportedOCL("empty invariant body")
        return bodies

    def encode_invariant(self, context: str, text: str, shared_vars: Dict,
                         scope: Dict, solver=None) -> List[Any]:
        """Return one Z3 formula per context slot:
        ``Implies(presence[i], <invariant evaluated for slot i>)``.

        Raises ``UnsupportedOCL`` if any construct cannot be encoded."""
        if context not in self.extractor.classes:
            raise UnsupportedOCL(f"unknown context class {context}")
        bodies = self.strip_header(text)
        asts = []
        for body in bodies:
            if body not in self._ast_cache:
                self._ast_cache[body] = OCLParser(body).parse()
            asts.append(self._ast_cache[body])

        self.shared_vars = shared_vars
        self.scope = scope
        self.solver = solver
        n = scope.get(f'n{context}', 5)
        presence = shared_vars[f'{context}_presence']
        formulas = []
        for i in range(n):
            env = {'self': ObjSet([(_TRUE, context, i)], single=True)}
            conj = []
            for ast in asts:
                val = self._eval(ast, env)
                conj.append(self._as_bool(val, "invariant body"))
            formulas.append(Implies(presence[i], _and(*conj)))
        return formulas

    # ── metamodel helpers ─────────────────────────────────────────────────

    def _n(self, cls: str) -> int:
        return self.scope.get(f'n{cls}', 5)

    def _presence(self, cls: str):
        return self.shared_vars[f'{cls}_presence']

    def _ancestors(self, cls: str) -> List[str]:
        chain = [cls]
        seen = {cls}
        cur = cls
        while True:
            parent = self.extractor.get_parent(cur)
            if not parent or parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
            cur = parent
        return chain

    def _find_attribute(self, cls: str, name: str):
        for c in self._ancestors(cls):
            a = self.extractor.get_attribute_by_name(c, name)
            if a is not None:
                return c, a
        return None, None

    def _find_reference(self, cls: str, name: str):
        for c in self._ancestors(cls):
            a = self.extractor.get_association_by_ref(c, name)
            if a is not None:
                return c, a
        return None, None

    def _attr_vars(self, cls: str, name: str) -> List[Any]:
        key = f'{cls}.{name}'
        vars_ = self.shared_vars.get(key)
        if vars_ is not None:
            return vars_
        decl_cls, attr = self._find_attribute(cls, name)
        if attr is None:
            raise UnsupportedOCL(f"unknown feature {cls}.{name}")
        # Inherited attribute: materialise per-slot variables for *cls* lazily,
        # with the same domain bounds the checker uses for declared attributes.
        z3_type = self.checker._get_z3_type(attr.attr_type, name)
        n = self._n(cls)
        vars_ = [z3_type(f"{cls}_{i}_{name}") for i in range(n)]
        self.shared_vars[key] = vars_
        if self.solver is not None:
            presence = self._presence(cls)
            if 'Int' in attr.attr_type:
                for i in range(n):
                    self.solver.add(Implies(presence[i], And(vars_[i] >= -1000, vars_[i] <= 1000000)))
            elif any(t in attr.attr_type for t in ('Real', 'Float', 'Double')):
                for i in range(n):
                    self.solver.add(Implies(presence[i], And(vars_[i] >= 0, vars_[i] <= 1000000)))
        return vars_

    def _ref_elems(self, g, cls: str, idx: int, name: str):
        """Navigate reference *name* from slot (cls, idx) guarded by *g*.
        Returns (elems, is_collection)."""
        decl_cls, assoc = self._find_reference(cls, name)
        if assoc is None:
            raise UnsupportedOCL(f"unknown feature {cls}.{name}")
        if decl_cls != cls:
            raise UnsupportedOCL(f"inherited reference {cls}.{name} (declared on {decl_cls})")
        target = assoc.target_class
        n_t = self._n(target)
        t_pres = self._presence(target)
        key = f'{cls}.{name}'
        if assoc.is_collection:
            R = self.shared_vars[key]
            return [(_and(g, R[idx][j], t_pres[j]), target, j) for j in range(n_t)], True
        ref = self.shared_vars[key]
        ref_present = self.shared_vars.get(f'{key}_present')
        elems = []
        for j in range(n_t):
            parts = [g, ref[idx] == j, t_pres[j]]
            if ref_present is not None:
                parts.append(ref_present[idx])
            elems.append((_and(*parts), target, j))
        return elems, False

    # ── value helpers ─────────────────────────────────────────────────────

    def _as_bool(self, v: Value, what: str):
        if isinstance(v, Scalar) and v.kind == 'bool':
            if v.alts:
                return self._distribute(v, lambda e: e)
            return _and(v.defined, v.expr) if v.defined is not None else v.expr
        raise UnsupportedOCL(f"{what} is not boolean ({type(v).__name__})")

    def _scalar_of_slot(self, cls: str, idx: int, name: str) -> Scalar:
        vars_ = self._attr_vars(cls, name)
        kind = _kind_of(vars_[idx])
        ecore_string = False
        if kind != 'string':
            _, attr = self._find_attribute(cls, name)
            ecore_string = attr is not None and 'String' in attr.attr_type
        return Scalar(vars_[idx], kind, ecore_string=ecore_string)

    def _string_form_vars(self, cls: str, name: str) -> List[Any]:
        """String-sorted view of a date-coded ``EString`` attribute.

        ``DateAdapter`` makes the checker Int-code date-like ``EString``
        attributes so that ordering comparisons work.  String operations
        (``size()``, ``concat``, ...) on such an attribute are evaluated on a
        separate String variable per slot.  The two views are not linked
        (Z3 cannot relate them), so a set that pins the Int view to a literal
        and constrains the String view contradictorily is under-approximated;
        this is the same limitation the legacy encoders have, minus the dropped
        sub-expressions.
        """
        from z3 import String
        key = f'{cls}.{name}__str'
        vars_ = self.shared_vars.get(key)
        if vars_ is None:
            vars_ = [String(f"{cls}_{i}_{name}__str") for i in range(self._n(cls))]
            self.shared_vars[key] = vars_
        return vars_

    def _string_form(self, node, env: Dict[str, Value]) -> Optional[Scalar]:
        """Evaluate ``node`` (a ``.attr`` navigation) using the String view of a
        date-coded EString attribute; None if not applicable."""
        if node[0] != 'nav':
            return None
        base = self._eval(node[1], env)
        if not isinstance(base, ObjSet) or not base.single or not base.elems:
            return None
        name = node[2]
        cls = base.elems[0][1]
        _, attr = self._find_attribute(cls, name)
        if attr is None or 'String' not in attr.attr_type:
            return None
        alts = []
        for g, c, i in base.elems:
            sv = self._string_form_vars(c, name)
            alts.append((g, Scalar(sv[i], 'string')))
        return self._collapse(alts, 'string')

    def _collapse(self, alts: List[Tuple[Any, Scalar]], kind: str) -> Scalar:
        """Turn guarded alternatives into one Scalar (If-chain)."""
        expr = _ite_chain([(g, s.expr) for g, s in alts], kind)
        defined_parts = []
        flat = []
        for g, s in alts:
            defined_parts.append(_and(g, s.defined) if s.defined is not None else g)
            if s.alts:
                flat.extend((_and(g, g2), e2) for g2, e2 in s.alts)
            else:
                flat.append((g, s.expr))
        return Scalar(expr, kind, defined=_or(defined_parts),
                      ecore_string=bool(alts) and all(s.ecore_string for _, s in alts),
                      alts=flat if flat else None)

    def _map_alts(self, s: Scalar, fn, kind: str, defined=None, **extra) -> Scalar:
        """Apply ``fn`` to a scalar's expression (per alternative when present)."""
        if s.alts:
            new_alts = [(g, fn(e)) for g, e in s.alts]
            return Scalar(_ite_chain(new_alts, kind), kind,
                          defined=s.defined if defined is None else defined,
                          alts=new_alts, **extra)
        return Scalar(fn(s.expr), kind, defined=s.defined if defined is None else defined, **extra)

    def _distribute(self, s: Scalar, pred):
        """``pred`` applied to a scalar as a Bool: Or_j(g_j and pred(e_j)) when it
        has alternatives (which also asserts definedness), else pred(expr)."""
        if s.alts:
            return _or([_and(g, pred(e)) for g, e in s.alts])
        return pred(s.expr)

    def _elems_of(self, v: Value) -> Tuple[List[Tuple[Any, Any]], str]:
        """Return (elements, kind) where kind is 'obj' or 'val'."""
        if isinstance(v, ObjSet):
            return [(g, (c, i)) for g, c, i in v.elems], 'obj'
        if isinstance(v, ValSet):
            return [(g, s) for g, s in v.elems], 'val'
        if isinstance(v, Scalar):
            return [(v.defined if v.defined is not None else _TRUE,
                     Scalar(v.expr, v.kind, None, v.literal, v.case))], 'val'
        if isinstance(v, NullVal):
            return [], 'val'
        raise UnsupportedOCL(f"not a collection ({type(v).__name__})")

    def _wrap(self, elems, kind: str, single: bool = False) -> Value:
        if kind == 'obj':
            return ObjSet([(g, c, i) for g, (c, i) in elems], single=single)
        return ValSet(elems)

    def _bind(self, kind: str, payload) -> Value:
        if kind == 'obj':
            c, i = payload
            return ObjSet([(_TRUE, c, i)], single=True)
        return payload

    # ── evaluation ────────────────────────────────────────────────────────

    def _eval(self, node, env: Dict[str, Value]) -> Value:
        tag = node[0]
        if tag == 'num':
            v = node[1]
            if isinstance(v, float):
                return Scalar(RealVal(v), 'real', literal=v)
            return Scalar(IntVal(v), 'int', literal=v)
        if tag == 'str':
            return Scalar(StringVal(node[1]), 'string', literal=node[1])
        if tag == 'bool':
            return Scalar(BoolVal(node[1]), 'bool', literal=node[1])
        if tag == 'null':
            return NullVal()
        if tag == 'enum':
            lit = node[1]
            return Scalar(IntVal(self.checker._string_to_int(lit)), 'int', literal=lit)
        if tag == 'var':
            name = node[1]
            if name in env:
                return env[name]
            if name in self.extractor.classes:
                return TypeRef(name)
            raise UnsupportedOCL(f"unbound identifier {name}")
        if tag == 'collit':
            items = [self._eval(it, env) for it in node[1]]
            elems = []
            for it in items:
                if isinstance(it, Scalar):
                    elems.append((it.defined if it.defined is not None else _TRUE, it))
                else:
                    raise UnsupportedOCL("collection literal of non-primitive values")
            return ValSet(elems)
        if tag == 'nav':
            return self._eval_nav(self._eval(node[1], env), node[2])
        if tag == 'call':
            return self._eval_call(self._eval(node[1], env), node[2], node[3], env, node[1])
        if tag == 'arrow':
            return self._eval_arrow(self._eval(node[1], env), node[2], node[3], node[4], node[5], env)
        if tag == 'un':
            return self._eval_unary(node[1], self._eval(node[2], env))
        if tag == 'bin':
            return self._eval_binary(node[1], self._eval(node[2], env), self._eval(node[3], env))
        if tag == 'if':
            return self._eval_if(self._eval(node[1], env), self._eval(node[2], env), self._eval(node[3], env))
        if tag == 'let':
            env2 = dict(env)
            env2[node[1]] = self._eval(node[2], env)
            return self._eval(node[3], env2)
        raise UnsupportedOCL(f"unknown AST node {tag}")

    # -- navigation ---------------------------------------------------------
    def _eval_nav(self, base: Value, name: str) -> Value:
        if isinstance(base, TypeRef):
            raise UnsupportedOCL(f"static feature {base.name}.{name}")
        if isinstance(base, NullVal):
            raise UnsupportedOCL("navigation from null")
        if isinstance(base, (Scalar, ValSet)):
            raise UnsupportedOCL(f"property {name} on a primitive value")
        assert isinstance(base, ObjSet)
        if not base.elems:
            return ObjSet([], single=base.single)
        cls = base.elems[0][1]
        if any(c != cls for _, c, _ in base.elems):
            raise UnsupportedOCL("heterogeneous object collection")
        decl, attr = self._find_attribute(cls, name)
        if attr is not None:
            alts = [(g, self._scalar_of_slot(c, i, name)) for g, c, i in base.elems]
            if base.single:
                return self._collapse(alts, alts[0][1].kind)
            return ValSet(alts)
        decl, assoc = self._find_reference(cls, name)
        if assoc is None:
            raise UnsupportedOCL(f"unknown feature {cls}.{name}")
        elems: List[Tuple[Any, str, int]] = []
        for g, c, i in base.elems:
            sub, is_coll = self._ref_elems(g, c, i, name)
            elems.extend(sub)
        return ObjSet(elems, single=base.single and not assoc.is_collection)

    # -- method calls (``.name(args)``) ---------------------------------------
    def _eval_call(self, base: Value, name: str, args: List[Any], env, base_node=None) -> Value:
        if isinstance(base, TypeRef):
            if name == 'allInstances' and not args:
                cls = base.name
                pres = self._presence(cls)
                return ObjSet([(pres[j], cls, j) for j in range(self._n(cls))], single=False)
            raise UnsupportedOCL(f"static operation {base.name}.{name}()")

        if isinstance(base, ObjSet):
            if name in ('oclIsTypeOf', 'oclIsKindOf'):
                if len(args) != 1 or args[0][0] != 'var':
                    raise UnsupportedOCL(f"{name} needs a class name")
                return self._type_test(base, args[0][1], exact=(name == 'oclIsTypeOf'))
            if name == 'oclAsType':
                return base
            if name == 'oclIsUndefined':
                return Scalar(Not(_or([g for g, _, _ in base.elems])), 'bool')
            if name == 'oclIsInvalid':
                return Scalar(_FALSE, 'bool')
            if name in ('size', 'isEmpty', 'notEmpty'):
                # ``.`` used where ``->`` was meant (common in generated OCL)
                return self._eval_arrow(base, name, [], [], None, env)
            raise UnsupportedOCL(f"operation {name}() on objects")

        if isinstance(base, ValSet):
            if name in ('size', 'isEmpty', 'notEmpty', 'sum'):
                return self._eval_arrow(base, name, [], [], None, env)
            raise UnsupportedOCL(f"operation {name}() on a collection of values")

        if isinstance(base, NullVal):
            if name == 'oclIsUndefined':
                return Scalar(_TRUE, 'bool')
            raise UnsupportedOCL(f"operation {name}() on null")

        assert isinstance(base, Scalar)
        argv = [self._eval(a, env) for a in args]
        if base.ecore_string and name in self._STRING_OPS:
            sf = self._string_form(base_node, env)
            if sf is not None:
                base = sf
        argv = [self._string_form(a, env) or v if (isinstance(v, Scalar) and v.ecore_string) else v
                for a, v in zip(args, argv)]
        return self._scalar_call(base, name, argv)

    _STRING_OPS = frozenset({'size', 'concat', 'indexOf', 'substring', 'matches', 'startsWith',
                             'endsWith', 'contains', 'equalsIgnoreCase', 'toUpperCase', 'toUpper',
                             'toLowerCase', 'toLower'})

    def _type_test(self, base: ObjSet, type_name: str, exact: bool) -> Scalar:
        if type_name not in self.extractor.classes:
            raise UnsupportedOCL(f"unknown class {type_name}")
        type_consts = self.shared_vars.get('__type_constants__', {})
        parts = []
        for g, cls, idx in base.elems:
            tau = self.shared_vars.get(f'{cls}_type')
            if exact:
                if tau is not None and type_name in type_consts:
                    parts.append(_and(g, tau[idx] == type_consts[type_name]))
                elif tau is None:
                    if cls == type_name:
                        parts.append(g)
            else:
                allowed = self.extractor.get_concrete_subtypes(type_name)
                if tau is not None:
                    opts = [tau[idx] == type_consts[cc] for cc in sorted(allowed) if cc in type_consts]
                    if opts:
                        parts.append(_and(g, _or(opts)))
                elif cls in self.extractor.get_all_subtypes(type_name) | {type_name}:
                    parts.append(g)
        return Scalar(_or(parts), 'bool')

    def _num(self, s: Scalar, what: str) -> Scalar:
        if not isinstance(s, Scalar) or s.kind not in self.NUMERIC or s.case is not None:
            raise UnsupportedOCL(f"{what} needs a numeric operand")
        return s

    def _str(self, s: Scalar, what: str) -> Scalar:
        if not isinstance(s, Scalar) or s.kind != 'string':
            raise UnsupportedOCL(f"{what} needs a string operand (Int-coded date fields are not strings)")
        return s

    def _scalar_call(self, base: Scalar, name: str, argv: List[Value]) -> Value:
        d = base.defined
        if name in ('toUpperCase', 'toUpper', 'toLowerCase', 'toLower') and not argv:
            self._str(base, name)
            if base.case is not None:
                raise UnsupportedOCL("nested case conversion")
            return Scalar(base.expr, 'string', d, None, 'upper' if 'Upper' in name else 'lower',
                          alts=base.alts)
        if name == 'size' and not argv:
            self._str(base, 'size()')
            return self._map_alts(base, Length, 'int')
        if name == 'concat' and len(argv) == 1:
            other = argv[0]
            self._str(base, 'concat')
            self._str(other, 'concat')
            if base.case is not None or other.case is not None:
                raise UnsupportedOCL("concat after case conversion")
            if other.alts:
                return Scalar(Concat(base.expr, other.expr), 'string', _and(d, other.defined))
            return self._map_alts(base, lambda e: Concat(e, other.expr), 'string', _and(d, other.defined))
        if name == 'indexOf' and len(argv) == 1:
            self._str(base, 'indexOf')
            other = self._str(argv[0], 'indexOf')
            if base.case is not None:
                raise UnsupportedOCL("indexOf after case conversion")
            # Repository convention: Z3 IndexOf (0-based, -1 when absent).
            return self._map_alts(base, lambda e: IndexOf(e, other.expr, IntVal(0)), 'int',
                                  _and(d, other.defined), indexof=(base, other.expr))
        if name == 'substring' and len(argv) == 2:
            self._str(base, 'substring')
            a, b = argv
            if not (isinstance(a, Scalar) and isinstance(b, Scalar)
                    and isinstance(a.literal, int) and isinstance(b.literal, int)):
                raise UnsupportedOCL("substring needs literal bounds")
            lo, hi = a.literal, b.literal
            if lo < 1 or hi < lo:
                return Scalar(StringVal(''), 'string', _FALSE)
            long_enough = self._distribute(base, lambda e: Length(e) >= hi)
            return self._map_alts(base, lambda e: SubString(e, lo - 1, hi - lo + 1), 'string',
                                  _and(d, long_enough))
        if name in ('matches', 'startsWith', 'endsWith', 'contains', 'equalsIgnoreCase') and len(argv) == 1:
            self._str(base, name)
            lit = argv[0]
            if not isinstance(lit, Scalar) or lit.kind != 'string':
                raise UnsupportedOCL(f"{name} needs a string argument")
            if name == 'matches':
                if lit.literal is None:
                    raise UnsupportedOCL("matches needs a literal pattern")
                rx = self.checker._ocl_regex_to_z3(lit.literal)
                return Scalar(_and(d, self._distribute(base, lambda e: InRe(e, rx))), 'bool')
            if name == 'equalsIgnoreCase':
                if lit.literal is None:
                    raise UnsupportedOCL("equalsIgnoreCase needs a literal")
                rx = self.checker._case_variant_regex_for_literal(lit.literal.lower(), 'lower')
                if rx is None:
                    return Scalar(_FALSE, 'bool')
                return Scalar(_and(d, self._distribute(base, lambda e: InRe(e, rx))), 'bool')
            fn = {'startsWith': lambda s_, l: PrefixOf(l, s_),
                  'endsWith': lambda s_, l: SuffixOf(l, s_),
                  'contains': lambda s_, l: Contains(s_, l)}[name]
            return Scalar(_and(d, lit.defined, self._distribute(base, lambda e: fn(e, lit.expr))), 'bool')
        if name == 'abs' and not argv:
            self._num(base, 'abs')
            return self._map_alts(base, lambda e: If(e >= 0, e, -e), base.kind)
        if name in ('max', 'min') and len(argv) == 1:
            self._num(base, name)
            o = self._num(argv[0], name)
            cond = base.expr >= o.expr if name == 'max' else base.expr <= o.expr
            kind = 'real' if 'real' in (base.kind, o.kind) else 'int'
            return Scalar(If(cond, base.expr, o.expr), kind, _and(d, o.defined))
        if name in ('floor', 'round') and not argv:
            self._num(base, name)
            if base.kind == 'int':
                return base
            fn = ToInt if name == 'floor' else (lambda e: ToInt(e + RealVal(0.5)))
            return self._map_alts(base, fn, 'int')
        if name == 'toReal' and not argv:
            self._num(base, name)
            if base.kind == 'real':
                return base
            return self._map_alts(base, ToReal, 'real')
        if name == 'oclIsUndefined' and not argv:
            return Scalar(Not(d) if d is not None else _FALSE, 'bool')
        if name == 'oclIsInvalid' and not argv:
            return Scalar(Not(d) if d is not None else _FALSE, 'bool')
        if name in ('isEmpty', 'notEmpty', 'size') and base.kind != 'string':
            raise UnsupportedOCL(f"{name}() on a non-collection")
        raise UnsupportedOCL(f"operation {name}/{len(argv)} on {base.kind}")

    # -- collection operations (``->name(...)``) -------------------------------
    def _eval_arrow(self, coll: Value, name: str, args: List[Any],
                    iters: List[str], body, env) -> Value:
        elems, ekind = self._elems_of(coll)
        single = isinstance(coll, ObjSet) and coll.single

        if name in ('asSet', 'asBag', 'asSequence', 'asOrderedSet', 'flatten'):
            return coll if not isinstance(coll, (Scalar, NullVal)) else self._wrap(elems, ekind)

        if name == 'size' and not args:
            return Scalar(Sum([If(g, 1, 0) for g, _ in elems]) if elems else IntVal(0), 'int',
                          card=[g for g, _ in elems])
        if name == 'isEmpty' and not args:
            return Scalar(Not(_or([g for g, _ in elems])), 'bool')
        if name == 'notEmpty' and not args:
            return Scalar(_or([g for g, _ in elems]), 'bool')

        if name in ('exists', 'forAll', 'one', 'select', 'reject', 'collect',
                    'collectNested', 'any', 'isUnique', 'sortedBy', 'closure', 'iterate'):
            if name in ('sortedBy', 'closure', 'iterate'):
                raise UnsupportedOCL(f"->{name}() is not supported")
            if body is None:
                if name == 'isUnique' and not args:
                    return self._is_unique(elems, ekind, None, None, env)
                if name in ('exists', 'forAll') and len(args) == 1:
                    # ->forAll(expr) with implicit iterator over a boolean collection
                    raise UnsupportedOCL(f"->{name}() without iterator")
                raise UnsupportedOCL(f"->{name}() needs an iterator body")
            if name in ('exists', 'forAll'):
                return self._quantify(name, elems, ekind, iters, body, env)
            if len(iters) != 1:
                raise UnsupportedOCL(f"->{name}() with {len(iters)} iterators")
            var = iters[0]
            if name == 'isUnique':
                return self._is_unique(elems, ekind, var, body, env)
            if name in ('select', 'reject', 'any', 'one'):
                guarded = []
                for g, payload in elems:
                    env2 = dict(env)
                    env2[var] = self._bind(ekind, payload)
                    b = self._as_bool(self._eval(body, env2), f"->{name} body")
                    guarded.append((_and(g, b if name != 'reject' else Not(b)), payload))
                if name == 'one':
                    return Scalar(PbEq([(g, 1) for g, _ in guarded], 1) if guarded else _FALSE, 'bool')
                if name == 'any':
                    if ekind == 'obj':
                        return ObjSet([(g, c, i) for g, (c, i) in guarded], single=True)
                    return self._collapse(guarded, guarded[0][1].kind if guarded else 'int')
                return self._wrap(guarded, ekind, single=single)
            # collect / collectNested
            out_obj: List[Tuple[Any, str, int]] = []
            out_val: List[Tuple[Any, Scalar]] = []
            for g, payload in elems:
                env2 = dict(env)
                env2[var] = self._bind(ekind, payload)
                r = self._eval(body, env2)
                if isinstance(r, ObjSet):
                    out_obj.extend((_and(g, g2), c, i) for g2, c, i in r.elems)
                elif isinstance(r, ValSet):
                    out_val.extend((_and(g, g2), s) for g2, s in r.elems)
                elif isinstance(r, Scalar):
                    out_val.append((g, r))
                else:
                    raise UnsupportedOCL("collect body of unsupported type")
            if out_obj and out_val:
                raise UnsupportedOCL("collect mixing objects and values")
            if out_obj or not out_val:
                return ObjSet(out_obj, single=False)
            return ValSet(out_val)

        if name == 'sum' and not args:
            if ekind != 'val':
                raise UnsupportedOCL("->sum() over objects")
            if not elems:
                return Scalar(IntVal(0), 'int')
            for _, s in elems:
                self._num(s, 'sum')
            kind = 'real' if any(s.kind == 'real' for _, s in elems) else 'int'
            total = Sum([If(g, s.expr, _default_for(kind)) for g, s in elems])
            defined = _and(*[Implies(g, s.defined) for g, s in elems if s.defined is not None])
            return Scalar(total, kind, defined)

        if name in ('includes', 'excludes', 'count') and len(args) == 1:
            x = self._eval(args[0], env)
            hits = self._membership(elems, ekind, x)
            if name == 'count':
                return Scalar(Sum([If(h, 1, 0) for h in hits]) if hits else IntVal(0), 'int', card=hits)
            m = _or(hits)
            return Scalar(m if name == 'includes' else Not(m), 'bool')

        if name in ('includesAll', 'excludesAll') and len(args) == 1:
            other, okind = self._elems_of(self._eval(args[0], env))
            parts = []
            for g2, p2 in other:
                hit = _or(self._membership(elems, ekind, self._bind(okind, p2)))
                parts.append(Implies(g2, hit if name == 'includesAll' else Not(hit)))
            return Scalar(_and(*parts), 'bool')

        if name in ('union', 'including', 'intersection', 'excluding', 'symmetricDifference') and len(args) == 1:
            other_v = self._eval(args[0], env)
            other, okind = self._elems_of(other_v)
            if other and elems and okind != ekind:
                raise UnsupportedOCL(f"->{name}() mixing objects and values")
            kind = ekind if elems else okind
            if name in ('union', 'including'):
                return self._wrap(elems + other, kind)
            if name == 'intersection':
                return self._wrap([(_and(g, _or(self._membership(other, kind, self._bind(kind, p)))), p)
                                   for g, p in elems], kind)
            if name == 'excluding':
                return self._wrap([(_and(g, Not(_or(self._membership(other, kind, self._bind(kind, p))))), p)
                                   for g, p in elems], kind)
            a_minus_b = [(_and(g, Not(_or(self._membership(other, kind, self._bind(kind, p))))), p) for g, p in elems]
            b_minus_a = [(_and(g, Not(_or(self._membership(elems, kind, self._bind(kind, p))))), p) for g, p in other]
            return self._wrap(a_minus_b + b_minus_a, kind)

        if name in ('max', 'min') and not args:
            if ekind != 'val' or not elems:
                raise UnsupportedOCL(f"->{name}() over objects or empty collection")
            for _, s in elems:
                self._num(s, name)
            kind = 'real' if any(s.kind == 'real' for _, s in elems) else 'int'
            # m = the element that is >= (<=) every other included element.
            # Encoded as an If-chain over "first extremal included element".
            alts = []
            for k, (g, s) in enumerate(elems):
                others = [Implies(g2, s.expr >= s2.expr if name == 'max' else s.expr <= s2.expr)
                          for k2, (g2, s2) in enumerate(elems) if k2 != k]
                alts.append((_and(g, *others), s))
            return self._collapse(alts, kind)

        raise UnsupportedOCL(f"->{name}/{len(args)} is not supported")

    def _quantify(self, name: str, elems, ekind: str, iters: List[str], body, env) -> Scalar:
        if not iters:
            raise UnsupportedOCL(f"->{name}() without iterator")
        parts = []

        def rec(depth: int, env_acc: Dict[str, Value], guard_acc):
            if depth == len(iters):
                b = self._as_bool(self._eval(body, env_acc), f"->{name} body")
                if name == 'exists':
                    parts.append(_and(guard_acc, b))
                else:
                    parts.append(Implies(guard_acc, b))
                return
            for g, payload in elems:
                env2 = dict(env_acc)
                env2[iters[depth]] = self._bind(ekind, payload)
                rec(depth + 1, env2, _and(guard_acc, g))

        rec(0, dict(env), _TRUE)
        if name == 'exists':
            return Scalar(_or(parts), 'bool')
        return Scalar(_and(*parts), 'bool')

    def _is_unique(self, elems, ekind, var, body, env) -> Scalar:
        vals: List[Tuple[Any, Scalar]] = []
        for g, payload in elems:
            if body is None:
                if ekind != 'val':
                    raise UnsupportedOCL("->isUnique() without body over objects")
                vals.append((g, payload))
            else:
                env2 = dict(env)
                env2[var] = self._bind(ekind, payload)
                r = self._eval(body, env2)
                if not isinstance(r, Scalar):
                    raise UnsupportedOCL("isUnique body must be a primitive value")
                vals.append((g, r))
        parts = []
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                ga, sa = vals[a]
                gb, sb = vals[b]
                ea, eb = self._coerce_pair(sa, sb)
                parts.append(Implies(_and(ga, gb), ea != eb))
        return Scalar(_and(*parts), 'bool')

    def _membership(self, elems, ekind: str, x: Value) -> List[Any]:
        """Guards under which *x* equals some element of the collection."""
        hits = []
        if ekind == 'obj':
            if isinstance(x, NullVal):
                return []
            if not isinstance(x, ObjSet):
                raise UnsupportedOCL("membership test of a value in an object collection")
            for g, (c, i) in elems:
                for gx, cx, ix in x.elems:
                    if cx == c and ix == i:
                        hits.append(_and(g, gx))
            return hits
        if not isinstance(x, Scalar):
            raise UnsupportedOCL("membership test of an object in a value collection")
        for g, s in elems:
            ea, eb = self._coerce_pair(s, x)
            hits.append(_and(g, s.defined, x.defined, ea == eb))
        return hits

    # -- unary / binary -----------------------------------------------------------
    def _eval_unary(self, op: str, v: Value) -> Value:
        if op == 'not':
            return Scalar(Not(self._as_bool(v, "not operand")), 'bool')
        if op == '-':
            s = self._num(v, 'unary minus')
            r = self._map_alts(s, lambda e: -e, s.kind)
            r.literal = -s.literal if s.literal is not None else None
            return r
        raise UnsupportedOCL(f"unary {op}")

    def _coerce_pair(self, a: Scalar, b: Scalar):
        """Make two scalars comparable; returns their Z3 expressions."""
        if a.kind == b.kind:
            return a.expr, b.expr
        if a.kind in self.NUMERIC and b.kind in self.NUMERIC:
            return a.expr, b.expr
        # Int-coded (date/enum) attribute against a string literal
        if a.kind == 'int' and b.kind == 'string' and b.literal is not None:
            return a.expr, IntVal(self.checker._string_to_int(b.literal))
        if b.kind == 'int' and a.kind == 'string' and a.literal is not None:
            return IntVal(self.checker._string_to_int(a.literal)), b.expr
        raise UnsupportedOCL(f"comparison between {a.kind} and {b.kind}")

    def _eval_binary(self, op: str, a: Value, b: Value) -> Value:
        if op in ('and', 'or', 'xor', 'implies'):
            ba = self._as_bool(a, f"'{op}' operand")
            bb = self._as_bool(b, f"'{op}' operand")
            if op == 'and':
                return Scalar(_and(ba, bb), 'bool')
            if op == 'or':
                return Scalar(Or(ba, bb), 'bool')
            if op == 'xor':
                return Scalar(Xor(ba, bb), 'bool')
            return Scalar(Implies(ba, bb), 'bool')

        if op in ('+', '-', '*', '/', 'div', 'mod'):
            sa = self._num(a, op)
            sb = self._num(b, op)
            defined = _and(sa.defined, sb.defined)
            kind = 'real' if 'real' in (sa.kind, sb.kind) else 'int'
            if op == '+':
                return Scalar(sa.expr + sb.expr, kind, defined)
            if op == '-':
                return Scalar(sa.expr - sb.expr, kind, defined)
            if op == '*':
                return Scalar(sa.expr * sb.expr, kind, defined)
            nz = sb.expr != 0
            if op == '/':
                ea = ToReal(sa.expr) if sa.kind == 'int' else sa.expr
                eb = ToReal(sb.expr) if sb.kind == 'int' else sb.expr
                return Scalar(ea / eb, 'real', _and(defined, nz))
            if kind != 'int':
                raise UnsupportedOCL(f"{op} on real operands")
            if op == 'div':
                return Scalar(sa.expr / sb.expr, 'int', _and(defined, nz))
            return Scalar(sa.expr % sb.expr, 'int', _and(defined, nz))

        if op in _REL_OPS:
            return self._compare(op, a, b)
        raise UnsupportedOCL(f"binary {op}")

    def _compare(self, op: str, a: Value, b: Value) -> Scalar:
        neg = op == '<>'
        # null comparisons
        if isinstance(a, NullVal) and isinstance(b, NullVal):
            return Scalar(BoolVal(not neg), 'bool')
        if isinstance(b, NullVal) or isinstance(a, NullVal):
            x = a if isinstance(b, NullVal) else b
            if op not in ('=', '<>'):
                raise UnsupportedOCL("ordering comparison with null")
            if isinstance(x, ObjSet):
                exists = _or([g for g, _, _ in x.elems])
                return Scalar(exists if neg else Not(exists), 'bool')
            if isinstance(x, Scalar):
                d = x.defined if x.defined is not None else _TRUE
                return Scalar(d if neg else Not(d), 'bool')
            if isinstance(x, ValSet):
                return Scalar(BoolVal(neg), 'bool')
            raise UnsupportedOCL("null comparison with unsupported operand")

        # object equality
        if isinstance(a, ObjSet) and isinstance(b, ObjSet):
            if op not in ('=', '<>'):
                raise UnsupportedOCL("ordering comparison between objects")
            same = []
            for ga, ca, ia in a.elems:
                for gb, cb, ib in b.elems:
                    if ca == cb and ia == ib:
                        same.append(_and(ga, gb))
            eq = _or(same)
            if not neg:
                return Scalar(eq, 'bool')
            both = _and(_or([g for g, _, _ in a.elems]), _or([g for g, _, _ in b.elems]))
            return Scalar(_and(both, Not(eq)), 'bool')

        if isinstance(a, (ObjSet, ValSet)) or isinstance(b, (ObjSet, ValSet)):
            raise UnsupportedOCL("comparison involving a collection")
        if not (isinstance(a, Scalar) and isinstance(b, Scalar)):
            raise UnsupportedOCL("comparison of unsupported operands")

        defined = _and(a.defined, b.defined)

        # case-converted string against a literal
        if a.case is not None or b.case is not None:
            s, lit = (a, b) if a.case is not None else (b, a)
            if op not in ('=', '<>') or lit.kind != 'string' or lit.literal is None or lit.case is not None:
                raise UnsupportedOCL("case conversion only supported against a string literal")
            rx = self.checker._case_variant_regex_for_literal(lit.literal, s.case)
            if rx is None:
                return Scalar(_and(defined, BoolVal(neg)), 'bool')
            m = self._distribute(s, lambda e: InRe(e, rx))
            if neg:
                return Scalar(_and(defined, Not(m)), 'bool')
            # with alternatives ``m`` already asserts definedness of ``s``
            return Scalar(_and(lit.defined if s.alts else defined, m), 'bool')

        # Cardinality against a literal: pseudo-Boolean constraint.
        if (a.card is not None) != (b.card is not None):
            c, lit = (a, b) if a.card is not None else (b, a)
            if lit.kind == 'int' and isinstance(lit.literal, int) and lit.literal == int(lit.literal):
                o = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}.get(op, op) if c is b else op
                pb = self._cardinality(c.card, o, int(lit.literal))
                if pb is not None:
                    return Scalar(_and(defined, pb), 'bool')

        # ``s.indexOf(t) >= 0`` and friends are membership tests; Z3's Contains
        # is far cheaper than reasoning about IndexOf.
        if (a.indexof is not None) != (b.indexof is not None):
            idx, lit = (a, b) if a.indexof is not None else (b, a)
            flipped = idx is b
            if isinstance(lit.literal, int) and lit.kind == 'int':
                o = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}.get(op, op) if flipped else op
                v = lit.literal
                hay, needle = idx.indexof
                found = self._distribute(hay, lambda e: Contains(e, needle))
                if (o, v) in (('>=', 0), ('>', -1), ('<>', -1)):
                    return Scalar(_and(defined, found), 'bool')
                if (o, v) in (('<', 0), ('<=', -1), ('=', -1)):
                    return Scalar(_and(defined, Not(found)), 'bool')

        ea, eb = self._coerce_pair(a, b)
        if a.kind == 'bool' or b.kind == 'bool':
            if op not in ('=', '<>'):
                raise UnsupportedOCL("ordering comparison between booleans")
        if a.kind == 'string' and b.kind == 'string':
            if op not in ('=', '<>'):
                raise UnsupportedOCL("ordering comparison between strings")
            # ``s <> ''`` / ``s = ''`` as length constraints: equivalent, and far
            # cheaper for Z3's sequence solver than disequality with a literal
            # when many string variables are involved.
            if b.literal == '' or a.literal == '':
                s_ = a if b.literal == '' else b
                pred = (lambda e: Length(e) > 0) if neg else (lambda e: Length(e) == 0)
                return Scalar(_and(defined, self._distribute(s_, pred)), 'bool')

        def cmp(x, y):
            if op == '=':
                return x == y
            if op == '<>':
                return x != y
            if op == '<':
                return x < y
            if op == '>':
                return x > y
            if op == '<=':
                return x <= y
            return x >= y

        # Distribute over navigation alternatives (see Scalar.alts).
        if a.alts and not b.alts:
            return Scalar(_and(defined, self._distribute(a, lambda e: cmp(e, eb))), 'bool')
        if b.alts and not a.alts:
            return Scalar(_and(defined, self._distribute(b, lambda e: cmp(ea, e))), 'bool')
        if a.alts and b.alts and len(a.alts) * len(b.alts) <= 64:
            pairs = [_and(ga, gb, cmp(x, y)) for ga, x in a.alts for gb, y in b.alts]
            return Scalar(_and(defined, _or(pairs)), 'bool')
        return Scalar(_and(defined, cmp(ea, eb)), 'bool')

    @staticmethod
    def _cardinality(guards: List[Any], op: str, n: int):
        """``|{g in guards : g}| op n`` as a pseudo-Boolean constraint."""
        guards = [g for g in guards if not (is_bool(g) and g.eq(_FALSE))]
        m = len(guards)

        def at_least(k):
            if k <= 0:
                return _TRUE
            if k > m:
                return _FALSE
            return AtLeast(*guards, k) if m > 1 else guards[0]

        def at_most(k):
            if k < 0:
                return _FALSE
            if k >= m:
                return _TRUE
            return AtMost(*guards, k) if m > 1 else Not(guards[0])

        if op == '>=':
            return at_least(n)
        if op == '>':
            return at_least(n + 1)
        if op == '<=':
            return at_most(n)
        if op == '<':
            return at_most(n - 1)
        if op == '=':
            return _and(at_least(n), at_most(n))
        if op == '<>':
            return Not(_and(at_least(n), at_most(n)))
        return None

    def _eval_if(self, c: Value, t: Value, e: Value) -> Value:
        cond = self._as_bool(c, "if condition")
        if isinstance(t, Scalar) and isinstance(e, Scalar):
            if t.kind != e.kind and not (t.kind in self.NUMERIC and e.kind in self.NUMERIC):
                raise UnsupportedOCL("if branches of different types")
            kind = 'real' if 'real' in (t.kind, e.kind) else t.kind
            et, ee = t.expr, e.expr
            if kind == 'real':
                et = ToReal(et) if t.kind == 'int' else et
                ee = ToReal(ee) if e.kind == 'int' else ee
            dt = t.defined if t.defined is not None else _TRUE
            de = e.defined if e.defined is not None else _TRUE
            return Scalar(If(cond, et, ee), kind, If(cond, dt, de))
        if isinstance(t, ObjSet) and isinstance(e, ObjSet):
            elems = [(_and(cond, g), c_, i) for g, c_, i in t.elems] + \
                    [(_and(Not(cond), g), c_, i) for g, c_, i in e.elems]
            return ObjSet(elems, single=t.single and e.single)
        raise UnsupportedOCL("if-then-else over unsupported branch types")
