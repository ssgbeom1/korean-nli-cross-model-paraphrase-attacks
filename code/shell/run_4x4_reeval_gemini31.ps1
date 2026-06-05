$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$python = "C:\Users\qjatj\Desktop\Paperproject\manufacturing_ai_llm_paper\.venv\Scripts\python.exe"
$geminiModel = "gemini-3.1-pro-preview"

$modelFileKey = @{
    clova = "hyperclova_x"
    gemini = "gemini"
    openai = "gpt"
    sonnet = "claude_sonnet"
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "===== $Name ====="
    $attempt = 1
    while ($attempt -le 2) {
        Write-Host "Attempt $attempt"
        & $python @Arguments
        if ($LASTEXITCODE -eq 0) {
            return
        }
        $attempt += 1
        Start-Sleep -Seconds 10
    }
    throw "Step failed after retries: $Name"
}

$sharedReference = "data/04_shared_valid_set/hyperclova_x_shared_valid_bert80.csv"

Write-Host "============================================"
Write-Host "Cache original predictions on shared set"
Write-Host "============================================"

foreach ($tgt in @("clova", "gemini", "openai", "sonnet")) {
    $tgtFile = $modelFileKey[$tgt]
    $argsList = @(
        "code/pipeline/06_evaluate_attacks.py",
        "--cache_originals",
        "--input", $sharedReference,
        "--target", $tgt,
        "--save_original_cache", "results/01_cross_model_evaluation/original_predictions/${tgtFile}_original_predictions.csv"
    )
    if ($tgt -eq "gemini") {
        $argsList += @("--model", $geminiModel)
    }
    Invoke-PythonStep -Name "Cache originals: $tgt" -Arguments $argsList
}

Write-Host "============================================"
Write-Host "Evaluate 4x4 cells"
Write-Host "============================================"

foreach ($gen in @("clova", "gemini", "openai", "sonnet")) {
    foreach ($tgt in @("clova", "gemini", "openai", "sonnet")) {
        $genFile = $modelFileKey[$gen]
        $tgtFile = $modelFileKey[$tgt]
        $argsList = @(
            "code/pipeline/06_evaluate_attacks.py",
            "--input", "data/04_shared_valid_set/${genFile}_shared_valid_bert80.csv",
            "--output", "results/01_cross_model_evaluation/${genFile}_as_generator/to_${tgtFile}.csv",
            "--target", $tgt,
            "--original_cache", "results/01_cross_model_evaluation/original_predictions/${tgtFile}_original_predictions.csv",
            "--resume"
        )
        if ($tgt -eq "gemini") {
            $argsList += @("--model", $geminiModel)
        }
        Invoke-PythonStep -Name "Evaluate: $gen -> $tgt" -Arguments $argsList
    }
}

Write-Host ""
Write-Host "DONE 4x4 reevaluation"
