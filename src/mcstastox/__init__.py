# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .LoadFile import Data as Read
from .LoadFile import Transfer
from .NexusSource import (
    ChunkOperation,
    EventColumn,
    EventSource,
    EventTransform,
    SourceContext,
    event_source,
)
from .NexusTemplate import (
    Diagnostic,
    NexusTemplate,
    RetainedDataset,
    ValidationReport,
    create_empty_template,
    create_template,
)
from .NexusWrite import (
    DestinationResult,
    ProgressUpdate,
    SourceResult,
    WriteReport,
    write_nexus,
)
from .Sampling import SampledEventData, SamplingSettings, sample_event_chunks
