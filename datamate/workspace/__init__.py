"""One business, ready for the agent: typed tables + sales model + schema card, cached."""

from datamate.workspace.batch import WorkspaceRef, prepare_many, prepare_one
from datamate.workspace.build import Workspace, build_workspace
from datamate.workspace.cache import load_cached, load_or_build

__all__ = [
    "Workspace",
    "WorkspaceRef",
    "build_workspace",
    "load_cached",
    "load_or_build",
    "prepare_many",
    "prepare_one",
]
