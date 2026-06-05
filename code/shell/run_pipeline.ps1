$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$modelFileKey = @{
    clova = "hyperclova_x"
    gemini = "gemini"
    openai = "gpt"
    sonnet = "claude_sonnet"
}

Write-Host "============================================"
Write-Host "Step 1: Load Data (KLUE-NLI 3000 samples)"
Write-Host "============================================"
python code/pipeline/01_load_data.py --n 3000 --seed 42 --out data/01_sampled_source/klue_nli_validation_sample_3000.csv

Write-Host ""
Write-Host "============================================"
Write-Host "Step 2: Generate LLM Attacks"
Write-Host "============================================"
foreach ($gen in @("clova", "gemini", "openai", "sonnet")) {
    Write-Host "--- Generating: $gen ---"
    $genFile = $modelFileKey[$gen]
    python code/pipeline/02_generate_llm_attacks.py `
        --generator $gen `
        --input data/01_sampled_source/klue_nli_validation_sample_3000.csv `
        --output "data/02_generated_attacks/${genFile}_attacks_3000.csv" `
        --n 3000
}

Write-Host ""
Write-Host "============================================"
Write-Host "Step 3: Generate Baselines"
Write-Host "============================================"
foreach ($method in @("bert_attack", "backtranslation", "eda")) {
    Write-Host "--- Baseline: $method ---"
    python code/pipeline/03_generate_baselines.py `
        --method $method `
        --input data/01_sampled_source/klue_nli_validation_sample_3000.csv `
        --output "data/02_generated_attacks/${method}_attacks_3000.csv"
}

Write-Host ""
Write-Host "============================================"
Write-Host "Step 4: Validate"
Write-Host "============================================"
foreach ($gen in @("clova", "gemini", "openai", "sonnet")) {
    $genFile = $modelFileKey[$gen]
    python code/pipeline/04_validate_attacks.py `
        --input "data/02_generated_attacks/${genFile}_attacks_3000.csv" `
        --output "data/03_validated_attacks/${genFile}_valid_bert80.csv"
}
foreach ($method in @("backtranslation", "bert_attack", "eda")) {
    python code/pipeline/04_validate_attacks.py `
        --input "data/02_generated_attacks/${method}_attacks_3000.csv" `
        --output "data/03_validated_attacks/${method}_valid_bert80.csv"
}

Write-Host ""
Write-Host "============================================"
Write-Host "Step 5: Build Shared Valid Set"
Write-Host "============================================"
python code/pipeline/05_build_shared_valid.py

Write-Host ""
Write-Host "============================================"
Write-Host "Step 6: Cache Original Predictions"
Write-Host "============================================"
foreach ($tgt in @("clova", "gemini", "openai", "sonnet")) {
    Write-Host "--- Caching originals for target=$tgt ---"
    $tgtFile = $modelFileKey[$tgt]
    python code/pipeline/06_evaluate_attacks.py `
        --cache_originals `
        --target $tgt `
        --input data/04_shared_valid_set/hyperclova_x_shared_valid_bert80.csv `
        --save_original_cache "results/01_cross_model_evaluation/original_predictions/${tgtFile}_original_predictions.csv"
}

Write-Host ""
Write-Host "============================================"
Write-Host "Step 7: 4x4 Cross Evaluation"
Write-Host "============================================"
foreach ($gen in @("clova", "gemini", "openai", "sonnet")) {
    foreach ($tgt in @("clova", "gemini", "openai", "sonnet")) {
        Write-Host "--- $gen -> $tgt ---"
        $genFile = $modelFileKey[$gen]
        $tgtFile = $modelFileKey[$tgt]
        python code/pipeline/06_evaluate_attacks.py `
            --input "data/04_shared_valid_set/${genFile}_shared_valid_bert80.csv" `
            --output "results/01_cross_model_evaluation/${genFile}_as_generator/to_${tgtFile}.csv" `
            --target $tgt `
            --original_cache "results/01_cross_model_evaluation/original_predictions/${tgtFile}_original_predictions.csv"
    }
}

Write-Host ""
Write-Host "============================================"
Write-Host "Step 8: 4x4 Analysis and Integrity"
Write-Host "============================================"
python code/analysis/02_verify_4x4_integrity.py --expected_rows 2209
python code/analysis/03_generate_4x4_analysis.py
python code/analysis/04_statistical_analysis_summary.py

Write-Host ""
Write-Host "============================================"
Write-Host "Step 9: Back Translation Comparison"
Write-Host "============================================"
python code/analysis/01_prepare_bt_shared.py

foreach ($tgt in @("clova", "gemini", "openai", "sonnet")) {
    $tgtFile = $modelFileKey[$tgt]
    python code/pipeline/06_evaluate_attacks.py `
        --input "data/04_shared_valid_set/backtranslation_shared_valid_bert80.csv" `
        --output "results/02_backtranslation_evaluation/to_${tgtFile}.csv" `
        --target $tgt `
        --original_cache "results/01_cross_model_evaluation/original_predictions/${tgtFile}_original_predictions.csv"
}

Write-Host ""
Write-Host "============================================"
Write-Host "Step 10: Quality Analysis"
Write-Host "============================================"
python code/analysis/05_calculate_quality_metrics.py
python code/analysis/06_generate_quality_figure.py

Write-Host ""
Write-Host "============================================"
Write-Host "Step 11: Optional Human-Adjusted ASR Artifacts"
Write-Host "============================================"
$humanMajorityResults = "results/05_human_evaluation/analysis/human_validation_majority_results.csv"
if (Test-Path $humanMajorityResults) {
    python code/analysis/12_compute_human_adjusted_asr.py
    python code/analysis/13_write_bt_llm_comparison_table.py
} else {
    Write-Host "Skipping human-adjusted ASR artifacts; missing $humanMajorityResults"
}

Write-Host ""
Write-Host "============================================"
Write-Host "DONE"
Write-Host "============================================"
