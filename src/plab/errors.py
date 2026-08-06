"""Exception types shared across the package."""


class DataError(Exception):
    """Raised when price data is missing, incomplete or unusable.

    Never caught internally to substitute a default: a gap in prices produces a wrong
    return, therefore a wrong Sharpe ratio.
    """
