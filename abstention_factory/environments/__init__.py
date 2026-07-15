"""Environment packages ship with the AgentAbstain dataset, not this repo.

Download the dataset from huggingface.co/datasets/antiquality/agentabstain
and point AGENTABSTAIN_DATA at it (default: ./data). The per-environment
packages under <AGENTABSTAIN_DATA>/environments then become importable as
abstention_factory.environments.<name>, which is what the registry expects.
"""
import os
from pathlib import Path

_env_dir = Path(os.environ.get("AGENTABSTAIN_DATA", "data")).resolve() / "environments"
if _env_dir.is_dir():
    __path__.append(str(_env_dir))
