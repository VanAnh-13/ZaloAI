"""Files of one business -> typed tables (no business-specific logic)."""

from datamate.ingest.build import build_business
from datamate.ingest.model import Business, ColumnInfo, Table

__all__ = ["Business", "ColumnInfo", "Table", "build_business"]
