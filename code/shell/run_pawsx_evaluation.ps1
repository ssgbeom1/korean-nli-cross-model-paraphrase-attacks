$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$python = "C:\Users\qjatj\Desktop\Paperproject\manufacturing_ai_llm_paper\.venv\Scripts\python.exe"
$inputFile = "data/05_external_benchmark/pawsx/pawsx_ko_en_balanced_sample.csv"
$outputDir = "results/06_external_benchmark/pawsx"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

function Invoke-PawsX {
    param(
        [string]$Name,
        [string]$Target,
        [string]$Output,
        [string[]]$ExtraArgs = @()
    )
    Write-Host ""
    Write-Host "===== PAWS-X -> $Name ====="
    $argsList = @(
        "code/analysis/15_evaluate_pawsx_external_benchmark.py",
        "--input", $inputFile,
        "--target", $Target,
        "--output", $Output,
        "--resume"
    ) + $ExtraArgs
    & $python @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "PAWS-X evaluation failed: $Name"
    }
}

Write-Host "============================================"
Write-Host "PAWS-X external benchmark evaluation"
Write-Host "============================================"

Invoke-PawsX -Name "Gemini" -Target "gemini" -Output "$outputDir/pawsx_ko_en_gemini.csv" -ExtraArgs @("--model", "gemini-3.1-pro-preview", "--sleep", "0.2")
Invoke-PawsX -Name "GPT" -Target "openai" -Output "$outputDir/pawsx_ko_en_gpt.csv" -ExtraArgs @("--sleep", "0.2")
Invoke-PawsX -Name "Claude Sonnet" -Target "sonnet" -Output "$outputDir/pawsx_ko_en_claude_sonnet.csv" -ExtraArgs @("--sleep", "0.2")
Invoke-PawsX -Name "HyperCLOVA X" -Target "clova" -Output "$outputDir/pawsx_ko_en_hyperclova_x.csv" -ExtraArgs @("--sleep", "1.0")

Write-Host ""
Write-Host "DONE PAWS-X external benchmark evaluation"
