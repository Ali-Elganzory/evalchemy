#!/usr/bin/env python3
"""
Generate individual evaluation scripts for each model-task combination.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import dotenv_values
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import ijson

# ---------------------------------------------------------------------------
# Static (non-cluster) configuration
# ---------------------------------------------------------------------------
BATCH_SIZE_CACHE_FILENAME = "batch_size_cache.json"
DISCOVERY_SCRIPTS_DIRNAME = "batch_size_discovery_scripts"
EVAL_SCRIPTS_DIRNAME = "eval_scripts"

# Layout under $EVALCHEMY_DIR. Used inside generated SLURM scripts so they can
# resolve the discovery CLI and shared cache file.
SCRIPTS_REL_DIR = "scripts"
DISCOVER_PYTHON_RELPATH = f"{SCRIPTS_REL_DIR}/discover_batch_size.py"
CACHE_FILE_RELPATH = f"{SCRIPTS_REL_DIR}/{BATCH_SIZE_CACHE_FILENAME}"
SLURM_LOGS_PREFIX = "slurm_logs/mv_exp"

SCRIPTS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = SCRIPTS_DIR.parent / "templates"
DISCOVERY_TEMPLATE_NAME = "batch_size_discovery.slurm.j2"
EVAL_TEMPLATE_NAME = "benchmark_evaluation.slurm.j2"

SUPPORTED_CLUSTERS = ("marenostrum", "jupiter", "leonardo")


# ---------------------------------------------------------------------------
# Cluster configuration loaded from per-cluster .env files
# ---------------------------------------------------------------------------
def _str_to_bool(value: str, *, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean from {key}={value!r}")


@dataclass(frozen=True)
class ClusterConfig:
    cluster: str
    slurm_account: str
    slurm_partition: str
    slurm_qos: str
    slurm_mail_user: str
    slurm_cpus_per_gpu: int
    discovery_slurm_max_time: str
    eval_slurm_max_time: str
    work_dir: str
    modules: tuple[str, ...]
    nccl_socket_ifname: str
    gloo_socket_ifname: str
    num_nodes: int
    gpus_per_node: int
    debug_short_dist_timeout: bool

    @classmethod
    def from_env(cls, cluster: str) -> "ClusterConfig":
        env_path = SCRIPTS_DIR / f"{cluster}.env"
        if not env_path.exists():
            raise FileNotFoundError(f"Cluster env file not found: {env_path}")

        raw = {k: v for k, v in dotenv_values(env_path).items() if v is not None}

        required = (
            "SLURM_ACCOUNT",
            "SLURM_PARTITION",
            "SLURM_QOS",
            "SLURM_MAIL_USER",
            "SLURM_CPUS_PER_GPU",
            "DISCOVERY_SLURM_MAX_TIME",
            "EVAL_SLURM_MAX_TIME",
            "WORK_DIR",
            "MODULES",
            "EVAL_NUM_NODES",
            "EVAL_GPUS_PER_NODE",
            "DEBUG_SHORT_DIST_TIMEOUT",
        )
        missing = [k for k in required if k not in raw]
        if missing:
            raise KeyError(f"Missing required keys in {env_path}: {', '.join(missing)}")

        modules = tuple(m for m in raw["MODULES"].split() if m)

        return cls(
            cluster=cluster,
            slurm_account=raw["SLURM_ACCOUNT"],
            slurm_partition=raw["SLURM_PARTITION"],
            slurm_qos=raw["SLURM_QOS"],
            slurm_mail_user=raw["SLURM_MAIL_USER"],
            slurm_cpus_per_gpu=int(raw["SLURM_CPUS_PER_GPU"]),
            discovery_slurm_max_time=raw["DISCOVERY_SLURM_MAX_TIME"],
            eval_slurm_max_time=raw["EVAL_SLURM_MAX_TIME"],
            work_dir=raw["WORK_DIR"],
            modules=modules,
            nccl_socket_ifname=raw.get("NCCL_SOCKET_IFNAME", ""),
            gloo_socket_ifname=raw.get("GLOO_SOCKET_IFNAME", ""),
            num_nodes=int(raw["EVAL_NUM_NODES"]),
            gpus_per_node=int(raw["EVAL_GPUS_PER_NODE"]),
            debug_short_dist_timeout=_str_to_bool(
                raw["DEBUG_SHORT_DIST_TIMEOUT"], key="DEBUG_SHORT_DIST_TIMEOUT"
            ),
        )


# Define the models (only uncommented/active models are used to generate scripts)
MODELS: list[str | tuple[str, int]] = [
    # ---------------------------------------------------------
    # Base Models
    # ---------------------------------------------------------
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16384-rope_theta-1M-long_sft_16k",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k",
    "ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate",
    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k",
    "ali-elganzory/SmolLM2-1.7B-WithChatTemplate",
    "ali-elganzory/SmolLM2-1.7B-16k",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-1.7B-Base",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k",
    "ali-elganzory/1.7b-MixtureVitae-100BT",
    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-WithChatTemplate",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate",
    "ali-elganzory/Baguettotron",
    "ali-elganzory/Baguettotron-longsft_16k",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k",
    # ---------------------------------------------------------
    # SFT Models (100% Finished)
    # ---------------------------------------------------------
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/Qwen2.5-1.5B-SFT-Tulu3-decontaminated",
    "ali-elganzory/Qwen3-1.7B-Base-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-SFT-Tulu3",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-SFT-Tulu3",
    "ali-elganzory/Baguettotron-SFT-Tulu3-decontaminated",
    "ali-elganzory/Baguettotron-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    # ---------------------------------------------------------
    # DPO Models (100% Finished)
    # ---------------------------------------------------------
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-DPO-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/Qwen2.5-1.5B-DPO-Tulu3-decontaminated",
    "ali-elganzory/Qwen3-1.7B-Base-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-DPO-Tulu3",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-DPO-Tulu3",
    "ali-elganzory/Baguettotron-DPO-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/Baguettotron-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    # ---------------------------------------------------------
    # Merged Models (100% Finished)
    # ---------------------------------------------------------
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-DPO-Tulu3-decontaminated",
    # ---------------------------------------------------------
    # OT3 Models (100% Finished)
    # ---------------------------------------------------------
    ("open-sci/sft__ot30k_1.7b-Comma0.1-300BT-longsft_16k-DPO-Tulu3-decontaminate", 32768),
    ("open-sci/sft__ot30k_1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k", 32768),
    ("open-sci/sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3", 32768),
    ("open-sci/sft__ot30k_open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_Qwen2.5-1.5B-DPO-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_Qwen2.5-1.5B-SFT-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_Qwen3-1.7B-Base-DPO-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_Qwen3-1.7B-Base-SFT-Tulu3-decontaminated", 32768),
    ("open-sci/sft__ot30k_SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated", 32768),
    # "open-sci/sft__ot30k_SmolLM2-1.7B-Instruct-16k",
    ("open-sci/sft__ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated", 32768),
    ("open-sci/sft_ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k_base", 32768),
]

# Define the tasks
TASKS = [
    "IFEval",
    "HumanEval",
    "MBPP",
    "AIME24",
    "AIME25",
    "AMC23",
    "gsm8k",
    "MATH500",
    "LiveCodeBench",
    "GPQADiamond",
    "JEEBench",
]


def _multi_node_socket_exports_bash(cluster: ClusterConfig) -> str:
    """Optional export lines for NCCL / Gloo socket interface (empty constants = no lines)."""
    lines: list[str] = []
    if cluster.nccl_socket_ifname.strip():
        lines.append(
            f'export NCCL_SOCKET_IFNAME="{cluster.nccl_socket_ifname.strip()}"'
        )
    if cluster.gloo_socket_ifname.strip():
        lines.append(
            f'export GLOO_SOCKET_IFNAME="{cluster.gloo_socket_ifname.strip()}"'
        )
    return ("\n".join(lines) + "\n") if lines else ""


def _multi_node_debug_timeout_bash(cluster: ClusterConfig) -> str:
    if cluster.debug_short_dist_timeout:
        return (
            "\n# Shorter distributed timeouts for faster feedback while debugging (not for production).\n"
            "export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=0:03:0\n"
            "export TORCH_DISTRIBUTED_DEBUG=DETAIL\n"
        )
    return (
        "\n# Debug: uncomment for faster rendezvous failure or richer dist logs (verify names for your PyTorch build).\n"
        "# export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=0:03:0\n"
        "# export TORCH_DISTRIBUTED_DEBUG=DETAIL\n"
        "# torchrun also exposes --rdzv_timeout (seconds) if you bypass Accelerate.\n"
    )


_jinja_env: Environment | None = None


def _get_jinja_env() -> Environment:
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=False,
            lstrip_blocks=False,
            undefined=StrictUndefined,
        )
    return _jinja_env


def get_batch_size_cache_path(script_dir: Path) -> Path:
    return script_dir / BATCH_SIZE_CACHE_FILENAME


def load_batch_size_cache(cache_path: Path) -> dict[str, dict[str, int]]:
    if not cache_path.exists():
        return {}

    try:
        raw_text = cache_path.read_text().strip()
        raw_cache = {} if not raw_text else json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in batch size cache {cache_path}: {e}") from e

    cache: dict[str, dict[str, int]] = {}
    for model, task_map in raw_cache.items():
        if not isinstance(model, str) or not isinstance(task_map, dict):
            continue

        validated_task_map: dict[str, int] = {}
        for task, batch_size in task_map.items():
            if isinstance(task, str) and isinstance(batch_size, int) and batch_size > 0:
                validated_task_map[task] = batch_size

        if validated_task_map:
            cache[model] = validated_task_map

    return cache


def resolve_cached_batch_size(
    batch_size_cache: dict[str, dict[str, int]], model: str, task: str
) -> int | None:
    return batch_size_cache.get(model, {}).get(task)


def generate_discovery_script(model: str, task: str, cluster: ClusterConfig) -> str:
    model_short = model.split("/")[-1]
    job_name = f"discover_bs_{model_short}_{task}"

    template = _get_jinja_env().get_template(DISCOVERY_TEMPLATE_NAME)
    return template.render(
        job_name=job_name,
        model=model,
        model_short=model_short,
        task=task,
        slurm_logs_prefix=SLURM_LOGS_PREFIX,
        discover_python_relpath=DISCOVER_PYTHON_RELPATH,
        cache_file_relpath=CACHE_FILE_RELPATH,
        slurm_max_time=cluster.discovery_slurm_max_time,
        slurm_partition=cluster.slurm_partition,
        slurm_qos=cluster.slurm_qos,
        slurm_account=cluster.slurm_account,
        slurm_cpus_per_gpu=cluster.slurm_cpus_per_gpu,
        slurm_mail_user=cluster.slurm_mail_user,
        work_dir=cluster.work_dir,
        modules=list(cluster.modules),
    )


def generate_script(
    model: str,
    task: str,
    cached_batch_size: int | str | None,
    cluster: ClusterConfig,
) -> str:
    """Generate the full script content for a model-task combination."""

    model_short = model.replace("/", "_").replace(" ", "_")
    job_name = f"eval_{model_short}_{task}"

    if cached_batch_size is None:
        cached_batch_size = "auto"

    template = _get_jinja_env().get_template(EVAL_TEMPLATE_NAME)
    return template.render(
        job_name=job_name,
        model=model,
        model_short=model_short,
        task=task,
        num_nodes=cluster.num_nodes,
        gpus_per_node=cluster.gpus_per_node,
        srun_cpus_per_task=cluster.slurm_cpus_per_gpu * cluster.gpus_per_node,
        cached_batch_size=cached_batch_size,
        socket_exports=_multi_node_socket_exports_bash(cluster),
        debug_timeout_block=_multi_node_debug_timeout_bash(cluster),
        slurm_logs_prefix=SLURM_LOGS_PREFIX,
        slurm_max_time=cluster.eval_slurm_max_time,
        slurm_partition=cluster.slurm_partition,
        slurm_qos=cluster.slurm_qos,
        slurm_account=cluster.slurm_account,
        slurm_cpus_per_gpu=cluster.slurm_cpus_per_gpu,
        slurm_mail_user=cluster.slurm_mail_user,
        work_dir=cluster.work_dir,
        modules=list(cluster.modules),
    )


def get_safe_filename(model: str, task: str) -> str:
    """Generate a safe filename for the script."""
    model = model.replace("/", "_").replace(" ", "_")
    return f"eval_{model}_{task}.sh"


def remove_scripts(eval_scripts_dir: Path):
    """Remove all scripts in the evaluation scripts directory."""
    if eval_scripts_dir.exists():
        for file in eval_scripts_dir.glob("*.sh"):
            file.unlink()


def _unique_models_preserve_order(model_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for mid in model_ids:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def cache_models_with_transformers(model_ids: list[str]) -> None:
    """Download each model and tokenizer into the local HF/Transformers cache."""
    unique = _unique_models_preserve_order(model_ids)
    n = len(unique)
    num_digits = len(str(n)) if n else 1
    print(f"Caching {n} unique model(s) via Transformers (tokenizer + weights)...")
    print()
    for i, model_id in enumerate(unique, start=1):
        print(f"[{i:0{num_digits}d}/{n}] Caching: {model_id}")
        try:
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device_map="auto"
            )
        except Exception as e:
            print(
                f"ERROR: failed to cache {model_id}: {e}\n"
                "If this is a private repo, set HF_TOKEN or HUGGING_FACE_HUB_TOKEN.",
                file=sys.stderr,
            )
            sys.exit(1)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SLURM evaluation scripts for each model-task pair.",
        epilog=(
            "By default, each unique model in MODELS is downloaded into the Hugging Face / "
            "Transformers cache (needed when compute nodes use TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE). "
            "Use --no-download-models to only regenerate shell scripts. "
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for private repositories."
        ),
    )
    parser.add_argument(
        "--cluster",
        choices=SUPPORTED_CLUSTERS,
        default="marenostrum",
        help="Cluster name; selects which <cluster>.env file to load (default: marenostrum).",
    )
    parser.add_argument(
        "--cache-models",
        nargs="+",
        help="Cache models with Transformers.",
        default=[],
    )
    parser.add_argument(
        "--no-download-models",
        action="store_true",
        help="Skip tokenizer and AutoModelForCausalLM cache warming; only write eval_scripts.",
    )
    parser.add_argument(
        "--skip-uncached-eval-scripts",
        action="store_true",
        help="Do not generate evaluation scripts for model/task pairs missing a cached batch size.",
    )
    return parser.parse_args()


def get_first_kv(filename: Path) -> tuple[str, dict] | None:
    try:
        with open(filename, "r") as f:
            parser = ijson.kvitems(f, "")
            key, value = next(parser)
            return key, value
    except Exception as e:
        return None


def get_finished_scripts(results_dir: Path) -> set[str]:
    finished_scripts: set[str] = set()
    if not results_dir.exists():
        return finished_scripts

    def process_model_dir(d):
        tasks: set[str] = set()
        files = list(os.scandir(d))
        for f in files:
            kv = get_first_kv(Path(f.path))
            if kv is None:
                continue
            _, results = kv
            for task, metrics in results.items():
                if metrics == {}:
                    continue
                tasks.add(task)
        model = d.name.replace("__", "/", 1)
        return [(model, task) for task in tasks]

    all_model_dirs = list(os.scandir(results_dir))
    finished_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for processed in tqdm(
            executor.map(process_model_dir, all_model_dirs),
            desc="Loading finished scripts",
            total=len(all_model_dirs),
        ):
            finished_list.extend(processed)

    for model, task in finished_list:
        finished_scripts.add(get_safe_filename(model, task))

    return finished_scripts


def get_running_scripts() -> set[str]:
    running_scripts = subprocess.check_output(
        ["squeue", "--me", "-h", "-o", "%j"], text=True
    )
    return {s.strip() + ".sh" for s in running_scripts.split("\n") if s.strip()}


def main():
    args = parse_args()

    cluster_config = ClusterConfig.from_env(args.cluster)
    print(f"Cluster: {cluster_config.cluster}")

    if args.cache_models:
        cache_models_with_transformers(args.cache_models)
        exit(0)

    if not args.no_download_models:
        cache_models_with_transformers(MODELS)

    script_dir = Path(__file__).parent
    batch_size_cache_path = get_batch_size_cache_path(script_dir)
    batch_size_cache = load_batch_size_cache(batch_size_cache_path)
    batch_size_cache_path.touch(exist_ok=True)
    if batch_size_cache_path.stat().st_size == 0:
        batch_size_cache_path.write_text("{}\n")

    eval_scripts_dir = script_dir / EVAL_SCRIPTS_DIRNAME
    eval_scripts_dir.mkdir(exist_ok=True)
    discovery_scripts_dir = script_dir / DISCOVERY_SCRIPTS_DIRNAME
    discovery_scripts_dir.mkdir(exist_ok=True)

    print(f"Creating evaluation scripts in: {eval_scripts_dir}")
    print(f"Creating discovery scripts in: {discovery_scripts_dir}")
    print(f"Total models: {len(MODELS)}")
    print(f"Total tasks: {len(TASKS)}")
    n_scripts = len(MODELS) * len(TASKS)
    print(f"Total scripts: {n_scripts}")
    print()

    remove_scripts(eval_scripts_dir)
    remove_scripts(discovery_scripts_dir)

    running_scripts = get_running_scripts()
    finished_scripts = get_finished_scripts(Path("logs"))
    all_scripts = []
    generated_scripts = []

    script_count = 0
    discovery_script_count = 0
    skipped_eval_script_count = 0

    for model in MODELS:
        for task in TASKS:
            filename = get_safe_filename(model, task)
            all_scripts.append(filename)

            if filename in running_scripts:
                continue
            if filename in finished_scripts:
                continue

            cached_batch_size = resolve_cached_batch_size(batch_size_cache, model, task)

            if cached_batch_size is None:
                discovery_script_content = generate_discovery_script(
                    model, task, cluster_config
                )
                discovery_filepath = discovery_scripts_dir / filename
                with open(discovery_filepath, "w") as f:
                    f.write(discovery_script_content)
                os.chmod(discovery_filepath, 0o755)
                discovery_script_count += 1

            if args.skip_uncached_eval_scripts and cached_batch_size is None:
                skipped_eval_script_count += 1
                continue

            script_content = generate_script(
                model, task, cached_batch_size, cluster_config
            )
            filepath = eval_scripts_dir / filename
            with open(filepath, "w") as f:
                f.write(script_content)
            os.chmod(filepath, 0o755)
            generated_scripts.append(filename)
            script_count += 1

    running_scripts = [s for s in running_scripts if s in all_scripts]
    finished_scripts = [s for s in finished_scripts if s in all_scripts]

    print()
    print(f"Finished scripts: {len(finished_scripts)}")
    print(f"Running scripts: {len(running_scripts)}")
    print(f"Skipped scripts: {skipped_eval_script_count}")
    print(f"Generated scripts (remaining evaluations): {script_count}")


if __name__ == "__main__":
    main()
