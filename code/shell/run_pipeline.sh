#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

declare -A MODEL_FILE_KEY=(
    [clova]="hyperclova_x"
    [gemini]="gemini"
    [openai]="gpt"
    [sonnet]="claude_sonnet"
)

echo "============================================"
echo "Step 1: Load Data (KLUE-NLI 3000 samples)"
echo "============================================"
python code/pipeline/01_load_data.py --n 3000 --seed 42 --out data/01_sampled_source/klue_nli_validation_sample_3000.csv

echo ""
echo "============================================"
echo "Step 2: Generate LLM Attacks"
echo "============================================"
for gen in clova gemini openai sonnet; do
    echo "--- Generating: $gen ---"
    gen_file="${MODEL_FILE_KEY[$gen]}"
    python code/pipeline/02_generate_llm_attacks.py \
        --generator "$gen" \
        --input data/01_sampled_source/klue_nli_validation_sample_3000.csv \
        --output "data/02_generated_attacks/${gen_file}_attacks_3000.csv" \
        --n 3000
done

echo ""
echo "============================================"
echo "Step 3: Generate Baselines"
echo "============================================"
for method in bert_attack backtranslation eda; do
    echo "--- Baseline: $method ---"
    python code/pipeline/03_generate_baselines.py \
        --method "$method" \
        --input data/01_sampled_source/klue_nli_validation_sample_3000.csv \
        --output "data/02_generated_attacks/${method}_attacks_3000.csv"
done

echo ""
echo "============================================"
echo "Step 4: Validate"
echo "============================================"
for gen in clova gemini openai sonnet; do
    gen_file="${MODEL_FILE_KEY[$gen]}"
    python code/pipeline/04_validate_attacks.py \
        --input "data/02_generated_attacks/${gen_file}_attacks_3000.csv" \
        --output "data/03_validated_attacks/${gen_file}_valid_bert80.csv"
done
for method in backtranslation bert_attack eda; do
    python code/pipeline/04_validate_attacks.py \
        --input "data/02_generated_attacks/${method}_attacks_3000.csv" \
        --output "data/03_validated_attacks/${method}_valid_bert80.csv"
done

echo ""
echo "============================================"
echo "Step 5: Build Shared Valid Set"
echo "============================================"
python code/pipeline/05_build_shared_valid.py

echo ""
echo "============================================"
echo "Step 6: Cache Original Predictions"
echo "============================================"
for tgt in clova gemini openai sonnet; do
    echo "--- Caching originals for target=$tgt ---"
    tgt_file="${MODEL_FILE_KEY[$tgt]}"
    python code/pipeline/06_evaluate_attacks.py \
        --cache_originals \
        --target "$tgt" \
        --input data/04_shared_valid_set/hyperclova_x_shared_valid_bert80.csv \
        --save_original_cache "results/01_cross_model_evaluation/original_predictions/${tgt_file}_original_predictions.csv"
done

echo ""
echo "============================================"
echo "Step 7: 4x4 Cross Evaluation"
echo "============================================"
for gen in clova gemini openai sonnet; do
    for tgt in clova gemini openai sonnet; do
        echo "--- $gen -> $tgt ---"
        gen_file="${MODEL_FILE_KEY[$gen]}"
        tgt_file="${MODEL_FILE_KEY[$tgt]}"
        python code/pipeline/06_evaluate_attacks.py \
            --input "data/04_shared_valid_set/${gen_file}_shared_valid_bert80.csv" \
            --output "results/01_cross_model_evaluation/${gen_file}_as_generator/to_${tgt_file}.csv" \
            --target "$tgt" \
            --original_cache "results/01_cross_model_evaluation/original_predictions/${tgt_file}_original_predictions.csv"
    done
done

echo ""
echo "============================================"
echo "Step 8: 4x4 Analysis and Integrity"
echo "============================================"
python code/analysis/02_verify_4x4_integrity.py --expected_rows 2209
python code/analysis/03_generate_4x4_analysis.py
python code/analysis/04_statistical_analysis_summary.py

echo ""
echo "============================================"
echo "Step 9: Back Translation Comparison"
echo "============================================"
python code/analysis/01_prepare_bt_shared.py

for tgt in clova gemini openai sonnet; do
    tgt_file="${MODEL_FILE_KEY[$tgt]}"
    python code/pipeline/06_evaluate_attacks.py \
        --input "data/04_shared_valid_set/backtranslation_shared_valid_bert80.csv" \
        --output "results/02_backtranslation_evaluation/to_${tgt_file}.csv" \
        --target "$tgt" \
        --original_cache "results/01_cross_model_evaluation/original_predictions/${tgt_file}_original_predictions.csv"
done

echo ""
echo "============================================"
echo "Step 10: Quality Analysis"
echo "============================================"
python code/analysis/05_calculate_quality_metrics.py
python code/analysis/06_generate_quality_figure.py

echo ""
echo "============================================"
echo "Step 11: Optional Human-Adjusted ASR Artifacts"
echo "============================================"
human_majority_results="results/05_human_evaluation/analysis/human_validation_majority_results.csv"
if [[ -f "$human_majority_results" ]]; then
    python code/analysis/12_compute_human_adjusted_asr.py
    python code/analysis/13_write_bt_llm_comparison_table.py
else
    echo "Skipping human-adjusted ASR artifacts; missing $human_majority_results"
fi

echo ""
echo "============================================"
echo "DONE"
echo "============================================"
