param([switch]$SmokeTest)
& (Join-Path $PSScriptRoot 'launch45.ps1') -SmokeTest:$SmokeTest
exit $LASTEXITCODE
