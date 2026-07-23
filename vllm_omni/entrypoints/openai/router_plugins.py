# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-plugin hook: third-party routers mounted on the OpenAI app.

An entry point in the ``vllm_omni.router_plugins`` group resolves to a
``register(app: FastAPI) -> None`` callable that mounts whatever routers it
owns via ``app.include_router(...)``. The hook is transport- and
semantics-free: no plugin-specific type appears here, and a plugin reads its
own configuration from its own environment.

Loading is STRICT and fail-closed, deliberately diverging from
``vllm.plugins.load_plugins_by_group``, whose per-plugin ``try/except`` logs a
load failure and continues. A plugin can raise midway through its own
``include_router`` calls, leaving the app half-mutated with no transactional
unmount; the only state in which such routes are guaranteed never to serve is
the window before the HTTP server accepts connections. Both ``load()`` and the
``register(app)`` call therefore run outside any except-block, so the first
exception propagates and aborts startup. An operator installed a serving-route
plugin on purpose: a silent no-mount is worse than a failed boot.

Discovery, the ``VLLM_PLUGINS`` allowlist, and availability logging follow the
vLLM loader's semantics.

WARNING: ``omni_run_server_worker`` runs once per API worker process, so
router plugins are loaded once per worker. They must be designed to be loaded
multiple times without causing issues -- the same caveat that applies to
``vllm.plugins.load_general_plugins``.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

ROUTER_PLUGINS_GROUP = "vllm_omni.router_plugins"
"""Entry-point group whose members mount routers on the OpenAI app."""


def _allowed_plugin_names() -> list[str] | None:
    """Return the ``VLLM_PLUGINS`` allowlist, or ``None`` when unset.

    Returns:
        The allowlisted plugin names, or ``None`` when every discovered plugin
        is eligible.
    """
    # Imported at call time, not module import time: the allowlist is read when
    # the server builds its app, matching vLLM's own lazy env evaluation.
    import vllm.envs as envs

    allowed: list[str] | None = envs.VLLM_PLUGINS
    return allowed


def load_router_plugins(app: FastAPI) -> None:
    """Load and invoke every eligible ``vllm_omni.router_plugins`` entry point.

    Must be called after serving state is initialized on ``app.state`` and
    before the HTTP server accepts connections: plugins delegate to the
    serving objects they find on the app, and a failure here must abort
    startup while no route can yet serve traffic.

    Args:
        app: The fully-built application handed to each plugin's ``register``.

    Raises:
        Exception: Whatever a plugin's ``load()`` or ``register(app)`` raises,
            propagated unchanged so startup fails before serving.
    """
    discovered_plugins = entry_points(group=ROUTER_PLUGINS_GROUP)
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group %s found.", ROUTER_PLUGINS_GROUP)
        return

    allowed_plugins = _allowed_plugin_names()

    logger.info("Available plugins for group %s:", ROUTER_PLUGINS_GROUP)
    for plugin in discovered_plugins:
        logger.info("- %s -> %s", plugin.name, plugin.value)

    if allowed_plugins is None:
        logger.info("All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.")

    for plugin in discovered_plugins:
        if allowed_plugins is not None and plugin.name not in allowed_plugins:
            continue
        if allowed_plugins is not None:
            logger.info("Loading plugin %s", plugin.name)
        # Outside any except-block by contract: the first failure aborts
        # startup rather than leaving half-mounted routes to serve.
        register = plugin.load()
        register(app)
