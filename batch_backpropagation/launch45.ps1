param([switch]$SmokeTest)
$ErrorActionPreference = 'Stop'
try {
    $taskRoot = $PSScriptRoot
    $anacondaPython = 'D:\program\anaconda\python.exe'
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $anacondaPython) {
        $taskPython = $anacondaPython
    } elseif (Test-Path -LiteralPath $bundledPython) {
        $taskPython = $bundledPython
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $taskPython = (Get-Command python).Source
    } else {
        throw 'Python was not found. Install Python and requirements.txt dependencies.'
    }
    $env:PYTHONUTF8 = '1'
    $taskArguments = @((Join-Path $taskRoot 'exercise45.py'))
    if ($SmokeTest) { $taskArguments += '--smoke-test' }
    & $taskPython @taskArguments
    exit $LASTEXITCODE
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
