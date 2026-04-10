import ast
import math
from typing import Callable

SAFE_RATE_FUNCTION_NAMESPACE: dict = {
    "__builtins__": {},
    "math": math,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "pow": pow,
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_CALLS: frozenset[str] = frozenset(
    SAFE_RATE_FUNCTION_NAMESPACE.keys() - {"__builtins__", "math"}
)

_ALLOWED_ATTR_ROOTS: frozenset[str] = frozenset({"math"})

_FORBIDDEN_NODE_TYPES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)

def validate_rate_function(rate_function_code: str) -> None:
    # 1. Verifica sintaxe
    try:
        tree = ast.parse(rate_function_code, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"order_generator.rate_function possui sintaxe inválida: {exc}"
        ) from exc

    # 2. A expressão raiz deve ser uma lambda
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError(
            "order_generator.rate_function deve ser uma expressão lambda, "
            f"mas recebeu: {type(tree.body).__name__}"
        )

    # 3. Varre toda a árvore AST em busca de nós perigosos
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise ValueError(
                "order_generator.rate_function contém instrução proibida: "
                f"{type(node).__name__}"
            )

        # Bloqueia chamadas a funções fora do namespace seguro
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id not in _ALLOWED_CALLS:
                raise ValueError(
                    f"order_generator.rate_function chama função não permitida: '{func.id}'. "
                    f"Funções permitidas: {sorted(_ALLOWED_CALLS)}"
                )

        # Bloqueia acesso a atributos de objetos não autorizados (ex: os.system)
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id not in _ALLOWED_ATTR_ROOTS
            ):
                raise ValueError(
                    "order_generator.rate_function acessa atributo de objeto não permitido: "
                    f"'{node.value.id}.{node.attr}'"
                )


def build_rate_function(rate_function_code: str) -> Callable:
    compiled = compile(
        ast.parse(rate_function_code, mode="eval"),
        filename="<rate_function>",
        mode="eval",
    )
    return eval(compiled, SAFE_RATE_FUNCTION_NAMESPACE.copy())  # noqa: S307
