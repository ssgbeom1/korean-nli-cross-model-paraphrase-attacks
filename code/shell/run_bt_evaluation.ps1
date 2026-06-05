$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$python = "C:\Users\qjatj\Desktop\Paperproject\manufacturing_ai_llm_paper\.venv\Scripts\python.exe"
$geminiModel = "gemini-3.1-pro-preview"
$inputFile = "data/04_shared_valid_set/backtranslation_shared_valid_bert80.csv"

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
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

Write-Host "============================================"
Write-Host "Back Translation evaluation"
Write-Host "============================================"

foreach ($tgt in @("clova", "gemini", "openai", "sonnet")) {
    $tgtFile = $modelFileKey[$tgt]
    $argsList = @(
        "code/pipeline/06_evaluate_attacks.py",
        "--input", $inputFile,
        "--output", "results/02_backtranslation_evaluation/to_${tgtFile}.csv",
        "--target", $tgt,
        "--original_cache", "results/01_cross_model_evaluation/original_predictions/${tgtFile}_original_predictions.csv",
        "--resume"
    )
    if ($tgt -eq "gemini") {
        $argsList += @("--model", $geminiModel)
    }
    if ($tgt -eq "clova") {
        $argsList += @("--row_delay", "2.0")
    }
    Invoke-PythonStep -Name "Evaluate BT -> $tgt" -Arguments $argsList
}

Write-Host ""
Write-Host "DONE Back Translation evaluation"
