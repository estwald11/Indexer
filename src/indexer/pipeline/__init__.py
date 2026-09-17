"""Pipeline orchestration: the stages wired together, contracts enforced."""

from indexer.pipeline.build import Assembly, assemble, build_indexes
from indexer.pipeline.ingest import BuildResult, IngestionPipeline
from indexer.pipeline.query import QueryEngine
from indexer.pipeline.stores import FileArtifactStore, FileCache, JsonLedger, UnitStore

__all__ = [
    "Assembly",
    "BuildResult",
    "FileArtifactStore",
    "FileCache",
    "IngestionPipeline",
    "JsonLedger",
    "QueryEngine",
    "UnitStore",
    "assemble",
    "build_indexes",
]
