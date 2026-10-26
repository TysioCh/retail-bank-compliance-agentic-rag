class ComplianceRagError(Exception):
    """Base project error."""


class IngestionError(ComplianceRagError):
    """Document ingestion failed."""


class RetrievalError(ComplianceRagError):
    """Retrieval failed."""


class AgentExecutionError(ComplianceRagError):
    """Agent execution failed."""
