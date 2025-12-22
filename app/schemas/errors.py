"""Error response schemas for consistent error handling."""
from pydantic import BaseModel
from typing import Optional, Any, Dict, List


class ErrorResponse(BaseModel):
    """Standard error response format."""
    detail: str
    code: Optional[str] = None
    status_code: int
    errors: Optional[List[Dict[str, Any]]] = None  # For validation errors


def create_error_response(
    detail: str,
    status_code: int,
    code: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a standardized error response."""
    response = {
        "detail": detail,
        "status_code": status_code,
    }
    if code:
        response["code"] = code
    if errors:
        response["errors"] = errors
    return response

