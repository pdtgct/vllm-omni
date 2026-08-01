# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-neutral, engine-visible persistent-state capability."""

from .batch import (
    PersistentStateBatch,
    gather_persistent_state_batch,
    scatter_persistent_state_batch,
    validate_persistent_state_batch,
)
from .connector import (
    PersistentStateConnector,
    UnsupportedPersistentStateCapability,
)
from .manager import PersistentStateManager, StateBinding
from .profile import (
    validate_persistent_state_declarations,
    validate_persistent_state_profile,
)
from .spec import (
    PersistentStateDescriptor,
    PersistentStateLayerBase,
    PersistentStateSpec,
)
from .storage import (
    PersistentStateStorage,
    allocate_persistent_state_storage,
    initialize_persistent_state_slot,
    persistent_state_storage_from_raw,
)

__all__ = [
    "PersistentStateBatch",
    "PersistentStateConnector",
    "PersistentStateDescriptor",
    "PersistentStateLayerBase",
    "PersistentStateManager",
    "PersistentStateSpec",
    "PersistentStateStorage",
    "StateBinding",
    "UnsupportedPersistentStateCapability",
    "allocate_persistent_state_storage",
    "gather_persistent_state_batch",
    "initialize_persistent_state_slot",
    "persistent_state_storage_from_raw",
    "scatter_persistent_state_batch",
    "validate_persistent_state_batch",
    "validate_persistent_state_declarations",
    "validate_persistent_state_profile",
]
