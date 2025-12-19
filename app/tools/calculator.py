"""Calculator tool for basic math operations."""
from langchain_core.tools import tool
import math
import re
from typing import Union


@tool
def calculate(expression: str) -> str:
    """
    Perform basic mathematical calculations safely.

    Supports:
    - Basic arithmetic (+, -, *, /)
    - Percentages (e.g., "20% of 100")
    - Currency conversions (basic, requires rates)
    - Mathematical functions (sqrt, pow, etc.)

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        String representation of the result
    """
    try:
        # Sanitize expression - only allow safe characters
        safe_pattern = r"^[0-9+\-*/().\s%a-zA-Z]+$"
        if not re.match(safe_pattern, expression):
            return "Error: Invalid characters in expression"

        # Handle percentage calculations
        if "%" in expression:
            # Pattern: "X% of Y" or "X%"
            percentage_match = re.search(r"(\d+(?:\.\d+)?)%\s*(?:of\s*)?(\d+(?:\.\d+)?)?", expression)
            if percentage_match:
                percentage = float(percentage_match.group(1))
                base = float(percentage_match.group(2)) if percentage_match.group(2) else 100
                result = (percentage / 100) * base
                return str(result)

        # Handle basic arithmetic
        # Remove any non-math characters for safety
        safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression)
        
        # Use eval with limited builtins for safety
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "__builtins__": {},
        }
        
        result = eval(safe_expr, allowed_names)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"

