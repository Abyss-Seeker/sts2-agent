param(
    [Parameter(Mandatory=$true)][string]$Sts2DataDir,
    [Parameter(Mandatory=$true)][string]$GodotExe
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    dotnet build STS2AgentOverlay.csproj -c Release "-p:Sts2DataDir=$Sts2DataDir" -p:NuGetAudit=false
    if ($LASTEXITCODE -ne 0) { throw 'Mod compilation failed' }
    New-Item -ItemType Directory -Force dist | Out-Null
    & $GodotExe --headless --path $PSScriptRoot --log-file "$PSScriptRoot/dist/pack.log" --script pack.gd
    if ($LASTEXITCODE -ne 0) { throw 'PCK packaging failed' }
    Copy-Item -LiteralPath '.godot/mono/temp/bin/Release/STS2AgentOverlay.dll' -Destination dist
    Copy-Item -LiteralPath 'STS2AgentOverlay.json' -Destination dist
    Write-Output "Ready: $PSScriptRoot/dist"
} finally { Pop-Location }
