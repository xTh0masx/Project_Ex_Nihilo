"""Bootstrap helpers wiring together Phase 1 components with shared logging."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from DataFeed import DataFeed
from Logger import Logger, get_logger
from Storage import Storage

PipeLineT = TypeVar("PipeLineT")

def bootstrap_pipline(
    feed_factory: callable[..., DataFeed],
    storage_factory: Callable[..., Storage],
    pipeline_factory: Callable[..., PipeLineT],
    *,
    logger: Optional[Logger] = None,
    feed_kwargs: Optional[Dict[str, Any]] = None,
    storage_kwargs: Optional[Dict[str, Any]] = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[DataFeed, Storage, PipeLineT]:
    """Instantiate the core components with a shared :class: 'Logger' instance.

    Parameters
    -----------
    feed_factory:
        Callable returning a concrete :class: 'DataFeed' implementation.
    storage_factory:
        Callable returning a concrete :class: 'Storage' implementation.
    pipeline_factory:
        Callable returning the strategy or processing pipeline coordinating the feed and storage layers.
    logger:
    Optional :class: Logger instance. When ''None'' a default logger created by :func: get_logger is used.
    feed_kwargs, storage_kwargs, pipeline_kwargs:
        Optional dictionaries forward to the respective factory call.

    Returns
    ----------
    tuple
        The instantiated ''(feed, storage, pipeline)'' triple using the same logging context.
        """

    shared_logger = logger or get_logger(name="phase1.orchestration")

    feed = feed_factory(logger=shared_logger, **(feed_kwargs or {}))
    storage = storage_factory(logger=shared_logger, **(storage_kwargs or {}))
    pipeline = pipeline_factory(logger=shared_logger, **(pipeline_kwargs or {}))

    return feed, storage, pipeline
