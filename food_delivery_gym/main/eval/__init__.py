"""Utilitários de avaliação em lote e experimentos nomeados."""

from food_delivery_gym.main.eval.eval_parallel import (
    EpisodeJob,
    EvalJobSpec,
    create_eval_environment,
    derive_episode_seeds,
    run_episode_job,
    run_episodes_parallel,
)
from food_delivery_gym.main.eval.experiment import (
    DEFAULT_RUNS_ROOT,
    agent_dir,
    agent_complete,
    discover_agent_names,
    experiment_dir,
    finalize_run_json,
    load_yaml_spec,
    rebuild_summary_csv,
    resolve_agent_dir,
    resolve_experiment,
    write_run_json,
    write_summary_csv,
)
from food_delivery_gym.main.eval.silence import in_worker_process, silence_eval_noise

__all__ = [
    "DEFAULT_RUNS_ROOT",
    "EpisodeJob",
    "EvalJobSpec",
    "agent_dir",
    "agent_complete",
    "create_eval_environment",
    "derive_episode_seeds",
    "discover_agent_names",
    "experiment_dir",
    "finalize_run_json",
    "in_worker_process",
    "load_yaml_spec",
    "rebuild_summary_csv",
    "resolve_agent_dir",
    "resolve_experiment",
    "run_episode_job",
    "run_episodes_parallel",
    "silence_eval_noise",
    "write_run_json",
    "write_summary_csv",
]
