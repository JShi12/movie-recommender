"""Run the MovieLens recommender TFX pipeline locally."""

from __future__ import annotations

import argparse
import os
import platform
import re
from pathlib import Path

from tfx.orchestration.local.local_dag_runner import LocalDagRunner

import config
from pipeline_definition import create_pipeline


def patch_tfx_mlmd_windows_filtering_bug() -> None:
    """Patch a TFX/MLMD filter-query issue seen on some Windows installs."""
    if platform.system().lower() != "windows":
        return

    from tfx.orchestration.portable.mlmd import execution_lib

    if getattr(execution_lib, "_WINDOWS_FILTER_PATCHED", False):
        return

    def get_executions_associated_with_all_contexts_no_filter(metadata_handler, contexts):
        context_list = list(contexts)
        if not context_list:
            return []

        common_ids = None
        for context in context_list:
            executions = metadata_handler.store.get_executions_by_context(context.id)
            ids = {execution.id for execution in executions}
            common_ids = ids if common_ids is None else common_ids & ids
            if not common_ids:
                return []

        executions_by_id = {}
        for context in context_list:
            for execution in metadata_handler.store.get_executions_by_context(context.id):
                if execution.id in common_ids:
                    executions_by_id[execution.id] = execution
        return list(executions_by_id.values())

    execution_lib.get_executions_associated_with_all_contexts = (
        get_executions_associated_with_all_contexts_no_filter
    )
    execution_lib._WINDOWS_FILTER_PATCHED = True


def patch_tfx_windows_stateful_dir_bug() -> None:
    """Sanitize run-id suffixes used in stateful working directories on Windows."""
    if platform.system().lower() != "windows":
        return

    from tfx.orchestration.portable import outputs_utils

    if getattr(outputs_utils, "_WINDOWS_STATEFUL_DIR_PATCHED", False):
        return

    original = outputs_utils.get_stateful_working_directory

    def safe_get_stateful_working_directory(
        node_dir,
        execution_mode,
        pipeline_run_id="",
        execution_id=None,
    ):
        safe_run_id = re.sub(r'[<>:"/\\|?*]', "-", str(pipeline_run_id))
        return original(
            node_dir=node_dir,
            execution_mode=execution_mode,
            pipeline_run_id=safe_run_id,
            execution_id=execution_id,
        )

    outputs_utils.get_stateful_working_directory = safe_get_stateful_working_directory
    outputs_utils._WINDOWS_STATEFUL_DIR_PATCHED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-name", default=config.PIPELINE_NAME)
    parser.add_argument("--pipeline-root", type=Path, default=config.PIPELINE_ROOT)
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=config.METADATA_PATH)
    parser.add_argument("--serving-model-dir", type=Path, default=config.SERVING_MODEL_DIR)
    parser.add_argument("--transform-module", type=Path, default=config.TRANSFORM_MODULE_FILE)
    parser.add_argument("--trainer-module", type=Path, default=config.TRAINER_MODULE_FILE)
    parser.add_argument("--train-steps", type=int, default=config.TRAIN_STEPS)
    parser.add_argument("--eval-steps", type=int, default=config.EVAL_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_tfx_mlmd_windows_filtering_bug()
    patch_tfx_windows_stateful_dir_bug()

    tfx_pipeline = create_pipeline(
        pipeline_name=args.pipeline_name,
        pipeline_root=args.pipeline_root,
        data_root=args.data_root,
        transform_module_path=os.path.abspath(args.transform_module),
        trainer_module_path=os.path.abspath(args.trainer_module),
        serving_model_dir=args.serving_model_dir,
        metadata_path=args.metadata_path,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps,
    )

    print("=" * 80)
    print("RUNNING TFX PIPELINE")
    print("=" * 80)
    print(f"Pipeline: {tfx_pipeline.pipeline_info.pipeline_name}")
    print(f"Components: {len(tfx_pipeline.components)}")
    print(f"Data root: {args.data_root}")
    print(f"Pipeline root: {args.pipeline_root}")
    LocalDagRunner().run(tfx_pipeline)
    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
