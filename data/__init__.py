# Make data a package so configs can import data.dataset via LazyConfig.
# Re-export commonly used symbols for legacy imports.
from .dataset import DATA_TEST_ARGS, DATA_TEST_PATH  # noqa: F401
