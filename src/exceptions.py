class MemorySystemError(Exception):
    """Base exception for the memory system."""

    pass


class StateValidationError(MemorySystemError):
    """Raised when the session state is invalid or corrupted."""

    pass


class SessionRecoveryError(MemorySystemError):
    """Raised when a checkpoint fails to load and fallback is required."""

    pass


class VectorSearchError(MemorySystemError):
    """Raised when the Long-Term Memory (Vertex Search) is unavailable."""

    pass


class ConcurrentWriteError(MemorySystemError):
    """Raised when OCC detects a version conflict (another writer saved the checkpoint)."""

    pass
