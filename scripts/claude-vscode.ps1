[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("health", "models", "ask", "json")]
    [string]$Command,

    [string]$Prompt,

    [string]$SystemPrompt = "You are Claude Code in VS Code collaborating with Codex in the same workspace.",

    [string]$Model = "claude-sonnet",

    [string]$SchemaFile,

    [switch]$Raw
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$BridgeStatePath = Join-Path $WorkspaceRoot ".workspace_agents\vscode_claude_bridge.json"

function Get-BridgeState {
    if (-not (Test-Path $BridgeStatePath)) {
        throw "Bridge state file not found at $BridgeStatePath."
    }

    $state = Get-Content -Raw $BridgeStatePath | ConvertFrom-Json
    if (-not $state.host -or -not $state.port -or -not $state.token) {
        throw "Bridge state file is missing host, port, or token."
    }

    return $state
}

function Invoke-ClaudeBridge {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("GET", "POST")]
        [string]$Method,

        [Parameter(Mandatory = $true)]
        [string]$Path,

        [object]$Body
    )

    $state = Get-BridgeState
    $headers = @{
        "x-workspace-agents-bridge-token" = [string]$state.token
    }
    $uri = "http://{0}:{1}{2}" -f $state.host, $state.port, $Path

    try {
        if ($Method -eq "GET") {
            return Invoke-RestMethod -Uri $uri -Method Get -Headers $headers
        }

        $jsonBody = if ($null -eq $Body) { "{}" } else { $Body | ConvertTo-Json -Depth 50 }
        return Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -ContentType "application/json" -Body $jsonBody
    } catch {
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseText = $reader.ReadToEnd()
            throw "Claude bridge request failed: $responseText"
        }
        throw
    }
}

function Get-DefaultAnswerSchema {
    return @{
        type = "object"
        additionalProperties = $false
        properties = @{
            answer = @{
                type = "string"
            }
        }
        required = @("answer")
    }
}

switch ($Command) {
    "health" {
        $result = Invoke-ClaudeBridge -Method GET -Path "/health"
        $result | ConvertTo-Json -Depth 30
        break
    }

    "models" {
        $result = Invoke-ClaudeBridge -Method GET -Path "/health"
        if (-not $result.ok) {
            throw "Claude bridge health check returned ok=false."
        }
        $result.result.models | ConvertTo-Json -Depth 20
        break
    }

    "ask" {
        if ([string]::IsNullOrWhiteSpace($Prompt)) {
            throw "ask requires -Prompt."
        }

        $body = @{
            system_prompt = $SystemPrompt
            prompt = $Prompt
            schema = Get-DefaultAnswerSchema
            model = $Model
            phase = "ask"
        }

        $result = Invoke-ClaudeBridge -Method POST -Path "/generate-json" -Body $body
        if ($Raw) {
            $result | ConvertTo-Json -Depth 30
        } else {
            $result.result.payload.answer
        }
        break
    }

    "json" {
        if ([string]::IsNullOrWhiteSpace($Prompt)) {
            throw "json requires -Prompt."
        }
        if ([string]::IsNullOrWhiteSpace($SchemaFile)) {
            throw "json requires -SchemaFile."
        }
        if (-not (Test-Path $SchemaFile)) {
            throw "Schema file not found: $SchemaFile"
        }

        $schema = Get-Content -Raw $SchemaFile | ConvertFrom-Json
        $body = @{
            system_prompt = $SystemPrompt
            prompt = $Prompt
            schema = $schema
            model = $Model
            phase = "json"
        }

        $result = Invoke-ClaudeBridge -Method POST -Path "/generate-json" -Body $body
        $result | ConvertTo-Json -Depth 30
        break
    }
}
