"""Business meaning on top of the typed tables: column roles, order statuses, the
canonical sales model (SalesView) and rulebook ranking."""

from datamate.semantics.ranking import top_k
from datamate.semantics.roles import detect_roles
from datamate.semantics.sales import SalesView, build_sales
from datamate.semantics.status import CANCELLED, COMPLETED, RETURNED, SHIPPING, canonical_status

__all__ = [
    "CANCELLED",
    "COMPLETED",
    "RETURNED",
    "SHIPPING",
    "SalesView",
    "build_sales",
    "canonical_status",
    "detect_roles",
    "top_k",
]
