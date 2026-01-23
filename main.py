"""
main.py (updated): Paper-aligned FACTER pipeline with:
- Open-ended Top-K generation
- Catalog mapping (to handle non-catalog outputs)
- @10 metrics computed on mapped recommendations + Valid@10
- SNSR/SNSV proxy metrics over mapped rec lists
- CFR via counterfactual attribute flips (neutral system prompt)
- Zero-shot baseline (open-ended) with same mapping and metrics
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from facter.config import Config
from facter.data import DatasetLoader
from facter.models import load_models
from facter.fairness import ConformalFairnessValidator, _group_key
from facter.prompt_engine import FairPromptEngine
from facter.utils import setup_logging, generate_recommendations, evaluate_at_k_from_lists, evaluate_valid_at_k

from facter.catalog_map import CatalogMapper
from facter.metrics_fairness import compute_snsr_snsv, compute_cfr
from facter.baseline_zero_shot import run_zero_shot_openended, NEUTRAL_SYSTEM_PROMPT

from codecarbon import OfflineEmissionsTracker

import argparse
from pathlib import Path
import time
pd.set_option("display.max_colwidth", None) 
pd.set_option("display.max_columns", None)   
pd.set_option("display.width", 200)        


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_backbone", type=str, default=None,
        help=f"LLM (e.g. llama3, llama2, mistral)")
    parser.add_argument("--datasets_used", nargs="+", default=None, 
        help=f"List of datasets (e.g. ml-1m amazon)")
    parser.add_argument("--extract_dir", default=None)
    parser.add_argument("--preprocessed_path", default=None)
    parser.add_argument("--embedder_alt_public", type=str, default=None)

    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--top_k_recs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)

    parser.add_argument("--history_size", type=int, default=None)
    parser.add_argument("--min_seq_length", type=int, default=None)

    parser.add_argument("--protected_attributes", nargs="+", default=None)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lambda_fairness", type=float, default=None)
    parser.add_argument("--n_reference", type=int, default=None)
    parser.add_argument("--base_similarity", type=float, default=None)
    parser.add_argument("--mapping_similarity", type=float, default=None)

    parser.add_argument("--quantile_decay", type=float, default=None)
    parser.add_argument("--violation_memory_size", type=int, default=None)

    parser.add_argument("--min_group_size", type=int, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=None)

    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)

    return parser.parse_args()


def update_config_from_args(args):
    """
    Overrides Config values that are explicitly provided as arguments.
    """
    # Dataset selection (experiment-level)
    if args.datasets_used is not None:
        unknown = set(args.datasets_used) - set(Config.DATASETS.keys())
        if unknown:
            raise ValueError(
                f"Unknown datasets {unknown}. "
                f"Available: {list(Config.DATASETS.keys())}"
            )
        Config.DATASETS_USED = args.datasets_used

    # Path override
    if args.extract_dir is not None:
        Config.EXTRACT_DIR = Path(args.extract_dir)
        Config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    if args.preprocessed_path is not None:
        Config.PREPROCESSED_PATH = Path(args.preprocessed_path)
        Config.PREPROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    # For all other attributes
    for arg_name, arg_value in vars(args).items():
        if arg_name in {"datasets_used", "extract_dir"}:
            continue
        if arg_value is None:
            continue

        config_attr = arg_name.upper()
        if hasattr(Config, config_attr):
            setattr(Config, config_attr, arg_value)


def main():
    args = parse_args()
    update_config_from_args(args)

    tracker = OfflineEmissionsTracker(country_iso_code="NLD")
    tracker.start()

    logger = setup_logging()
  
    np.random.seed(Config.RANDOM_SEED)

    for attr in dir(Config):
        if attr.isupper():
            logger.info(f"  {attr}:\t{getattr(Config, attr)}")

    embedder, tokenizer, model = load_models(prefer_public_finetuned_embedder=True)
    logger.info(f"embedder:\t{embedder}")
    logger.info(f"tokenizer:\t{tokenizer}")
    logger.info(f"model:\t{model}")

    results = {}
    for dataset_name in Config.DATASETS_USED:
        logger.info(f"\n=== Running {dataset_name.upper()} ===")
        preprocessing_start = time.time()

        loader = DatasetLoader(dataset_name)

        sample_size = Config.DATASETS[dataset_name]['sample_size']
        df = loader.prepare_prompts().dropna().sample(n=sample_size, 
                                                      random_state=Config.RANDOM_SEED).reset_index(drop=True)
        df.to_csv(Config.PREPROCESSED_PATH / f"{dataset_name}_{sample_size}.csv", index=False)

        logger.info(f"df.shape:\t{df.shape}")

        # Stratify by full tuple for stable eval
        strata = df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1)
        logger.info(f"strata.shape:\t{strata.shape}")
        logger.info(f"strata.nunique():\t{strata.nunique()}")
        logger.info(f"strata[:5]:\n{strata[:5]}")

        df = df[strata.map(strata.value_counts()) >= 2].copy()
        logger.info(f"df.shape:\t{df.shape}")

        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=Config.RANDOM_SEED,
            stratify=df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1),
        )
        n_total = len(df)
        n_train = len(train_df)
        n_test = len(test_df)
        logger.info(f"Train: {n_train} samples ({100 * n_train / n_total:.1f}%)")
        logger.info(f"Test : {n_test} samples ({100 * n_test / n_total:.1f}%)")
        strata_train = train_df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1)
        strata_test = test_df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1)
        logger.info(f"strata_train.nunique():\t{strata_train.nunique()}")
        logger.info(f"strata_test.nunique():\t{strata_test.nunique()}")
        train_df.to_csv(Config.PREPROCESSED_PATH / f"{dataset_name}_{sample_size}_train.csv", index=False)
        test_df.to_csv(Config.PREPROCESSED_PATH / f"{dataset_name}_{sample_size}_test.csv", index=False)

        if Config.TRAIN_SIZE:
            train_df = train_df.iloc[: Config.TRAIN_SIZE].copy()  # DEBUG: use small subset
            test_df = test_df.iloc[: Config.TRAIN_SIZE // 2].copy()

            logger.info(f"Train MINI: {train_df.shape}")
            logger.info(f"Test MINI: {test_df.shape}")
        
        logger.info(f"train_df[:5]:\n{train_df[:5]}")
        logger.info(f"test_df[:5]:\n{test_df[:5]}")

        # Build catalog mapper
        mapper = CatalogMapper(embedder, loader.item_db)
        mapper.build(dedup=True)

        # -------------------------
        # Offline calibration (FASTER: use rank-1 from open-ended)
        # -------------------------
        logger.info("Calibration generation (open-ended Top-K)...")
        calib_start = time.time()
        cal_recs = generate_recommendations(train_df["prompt"].tolist(), system_msg="", tokenizer=tokenizer, model=model)
        logger.info(f"len(cal_recs):\t{len(cal_recs)}")
        logger.info(f"cal_recs[:5]:\n{cal_recs[:5]}")

        cal_groups = [
            _group_key({k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES})
            for _, row in train_df.iterrows()
        ]
        logger.info(f"len(cal_groups):\t{len(cal_groups)}")
        logger.info(f"cal_groups[:5]:\n{cal_groups[:5]}")

        validator = ConformalFairnessValidator(embedder, item_db=loader.item_db)
        validator.calibrate(
            cal_contexts=train_df["context"].tolist(),
            cal_prompts=train_df["prompt"].tolist(),
            cal_groups=cal_groups,
            cal_recs=cal_recs,
            cal_targets=train_df["target_title"].tolist(),
        )

        prompt_engine = FairPromptEngine(validator)

        # Helper for CFR generation (neutral)
        def generate_fn(prompts, system_msg):
            return generate_recommendations(prompts, system_msg, tokenizer, model)

        # -------------------------
        # Zero-shot baseline (task-matched open-ended)
        # -------------------------
        logger.info("Zero-shot baseline...")
        zero_shot_start = time.time()
        
        zs_raw = run_zero_shot_openended(test_df, tokenizer, model)
        logger.info(f"len(zs_raw):\t{len(zs_raw)}")
        logger.info(f"zs_raw[:5]:\n{zs_raw[:5]}")

        zs_map = []
        zs_valid = []
        for recs in zs_raw:
            mr = mapper.map_list(recs, k=Config.TOP_K_RECS, min_sim=Config.MAPPING_SIMILARITY)
            zs_map.append(mr.mapped_titles)
            zs_valid.append(mr.valid_at_k)
        logger.info(f"zs_map[:5]:\n{zs_map[:5]}")
        logger.info(f"zs_valid[:5]:\t{zs_valid[:5]}")

        zs_acc = evaluate_at_k_from_lists(zs_map, test_df["target_title"].tolist(), k=Config.TOP_K_RECS)
        zs_validm = evaluate_valid_at_k(zs_valid, k=Config.TOP_K_RECS)
        zs_sns = compute_snsr_snsv(test_df.assign(mapped_recs=zs_map), embedder, recs_col="mapped_recs", group_mode="tuple")
        logger.info("zs_cfr = compute_cfr...")
        zs_cfr = compute_cfr(
            test_df,
            embedder,
            generate_fn=generate_fn,
            system_msg_neutral=NEUTRAL_SYSTEM_PROMPT,
            k=Config.TOP_K_RECS,
            n_samples=min(200, len(test_df)),
            flip_mode="tuple",
            prompt_col="prompt",
        )

        # --- Zero-shot violations (no threshold updates) ---
        logger.info("Zero-shot violations (no threshold updates)...")
        zs_is_viol = []
        zs_scores = []

        for (_, row), mapped in zip(test_df.iterrows(), zs_map):
            attrs = {k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES}

            # compute S the same way validate() does, but without updating Q / memory
            yhat_title = mapped[0] if mapped else ""
            s = validator._score_S(
                context=row["context"],
                group=_group_key(attrs),
                y_hat_title=yhat_title,
                y_true_title=row["target_title"],
            )

            zs_scores.append(float(s))
            zs_is_viol.append(bool(s > validator.adaptive_threshold))

        zs_violation_count = int(np.sum(zs_is_viol))
        zs_violation_rate = float(np.mean(zs_is_viol)) if zs_is_viol else 0.0

        baseline_block = {
            "ZeroShot_OpenEnded": {
                **zs_acc,
                **zs_validm,
                "SNSR": zs_sns.SNSR,
                "SNSV": zs_sns.SNSV,
                "CFR": zs_cfr.CFR,
                "CFR_valid_rate": zs_cfr.valid_rate,
                "CFR_n_pairs": zs_cfr.n_pairs,
                "violation_count": zs_violation_count,
                "violation_rate": zs_violation_rate,
            }
        }
        logger.info(f"baseline_block:\n{baseline_block}")

        # -------------------------
        # FACTER iterations
        # -------------------------
        history = []
        total_inference_time = 0.0

        for it in range(1, Config.MAX_ITERATIONS+1):
            logger.info(f"Iteration {it} ...")
            iter_start = time.time()

            prompt_engine.set_iteration(it)

            facter_raw = []
            facter_mapped = []
            facter_valid = []
            is_viol = []
            scores = []
            thresholds = []

            for i, (_, row) in enumerate(test_df.iterrows()):
                attrs = {k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES}
                g = _group_key(attrs)

                system_msg = prompt_engine.generate_system_prompt(current_group=g)
                user_prompt = prompt_engine.update_prompt(row["prompt"], current_group=g)

                recs = generate_recommendations([user_prompt], system_msg, tokenizer, model)[0]
                # map
                mr = mapper.map_list(recs, k=Config.TOP_K_RECS, min_sim=Config.MAPPING_SIMILARITY)
                mapped = mr.mapped_titles

                if i < 5:
                    logger.info(f"system_msg\n{system_msg}")
                    logger.info(f"user_prompt\n{user_prompt}") 
                    logger.info(f"recs\n{recs}") 
                    logger.info(f"mr\n{mr}") 
                    logger.info(f"mapped\n{mapped}")

                v, s, q = validator.validate(
                    context=row["context"],
                    prompt=row["prompt"],
                    attrs=attrs,
                    recs=mapped,             # IMPORTANT: run validator on mapped titles
                    y_true_title=row["target_title"],
                )

                facter_raw.append(recs)
                facter_mapped.append(mapped)
                facter_valid.append(mr.valid_at_k)
                is_viol.append(v)
                scores.append(s)
                thresholds.append(q)
           
            eval_df = test_df.copy()
            eval_df["mapped_recs_raw"] = facter_raw
            eval_df["mapped_recs"] = facter_mapped
            eval_df["valid_at_k"] = facter_valid
            eval_df["is_violation"] = is_viol
            eval_df["S"] = scores
            eval_df["Q"] = thresholds
            eval_df.to_csv(Config.PREPROCESSED_PATH / f"{dataset_name}_eval_df_iter{it}.csv", index=False)

            logger.info(f"validator.violation_memory\n{validator.violation_memory}")
            logger.info(f"eval_df[:5]\n{eval_df[:5]}")

            viol_rate = float(np.mean(is_viol)) if is_viol else 0.0
            acc = evaluate_at_k_from_lists(facter_mapped, eval_df["target_title"].tolist(), k=Config.TOP_K_RECS)
            validm = evaluate_valid_at_k(facter_valid, k=Config.TOP_K_RECS)

            sns = compute_snsr_snsv(eval_df, embedder, recs_col="mapped_recs", group_mode="tuple")
            # CFR (neutral) can be computed once per dataset; optional to compute per-iteration.
            # Here we compute once in iteration 0 for speed; set to None otherwise.
            cfr = None
            if it == 1:
                cfr = compute_cfr(
                    eval_df,
                    embedder,
                    generate_fn=generate_fn,
                    system_msg_neutral=NEUTRAL_SYSTEM_PROMPT,
                    k=Config.TOP_K_RECS,
                    n_samples=min(200, len(eval_df)),
                    flip_mode="tuple",
                    prompt_col="prompt",
                )

            record = {
                "iteration": it,
                "violation_rate": viol_rate,
                "violation_count": int(np.sum(is_viol)),
                **acc,
                **validm,
                "SNSR": sns.SNSR,
                "SNSV": sns.SNSV,
                "Q_last": float(eval_df["Q"].iloc[-1]),
            }
            if cfr is not None:
                record.update({"CFR": cfr.CFR, "CFR_valid_rate": cfr.valid_rate, "CFR_n_pairs": cfr.n_pairs})

            logger.info(f"Iter {it}: {json.dumps(record, indent=2)}")
            total_inference_time += (time.time() - iter_start)
            history.append(record)

            if it >= 2 and viol_rate < 0.10:
                break

        results[dataset_name] = {
            "baseline": baseline_block,
            "history": history,
            "Q_alpha_init": float(validator.adaptive_threshold) if validator.adaptive_threshold is not None else None,
            "preprocessing_time": (calib_start - preprocessing_start)/60,
            "calib_time": (zero_shot_start - calib_start)/60,
            "total_inference_time": total_inference_time/60,

        }

    logger.info("\n=== FINAL RESULTS ===\n" + json.dumps(results, indent=2))

    tracker.stop()

    return results


if __name__ == "__main__":
    main()
