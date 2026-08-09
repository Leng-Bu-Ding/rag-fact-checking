from __future__ import annotations

import ast
import math
import operator

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("calculator exponent is too large")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError(f"unsupported calculator syntax: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate arithmetic without names, calls, attributes, or Python eval."""
    if not expression.strip():
        raise ValueError("calculator expression cannot be empty")
    if len(expression) > 200:
        raise ValueError("calculator expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid calculator expression") from error
    value = _evaluate(tree)
    if not math.isfinite(value):
        raise ValueError("calculator result must be finite")
    return value
