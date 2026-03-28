from __future__ import annotations

import ast
import math
import operator

try:
    import sympy  # type: ignore
except Exception:  # pragma: no cover
    sympy = None

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expr: str) -> float:
    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](eval_node(node.operand))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return float(eval_node(tree))


def verify_step(expression: str, claimed_result: float, tolerance: float = 1e-6) -> bool | None:
    """Return True/False if expression is parseable, else None for inconclusive."""
    try:
        if sympy is not None:
            actual = float(sympy.sympify(expression))
        else:
            actual = _safe_eval(expression)
    except Exception:
        return None
    return math.isclose(actual, claimed_result, rel_tol=tolerance, abs_tol=tolerance)


def verify_final(predicted: float | None, ground_truth: float, tolerance: float = 1e-6) -> bool:
    if predicted is None:
        return False
    return math.isclose(predicted, ground_truth, rel_tol=tolerance, abs_tol=tolerance)
