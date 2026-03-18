"""Package version helpers."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed package version, or a dev fallback in source mode."""
    try:
        return version("pytia")
    except PackageNotFoundError:
        return "0.1.0-dev"


__version__ = get_version()
