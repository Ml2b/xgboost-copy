[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("submit", "status", "diff", "apply", "list", "config")]
    [string]$Command,

    [string]$Prompt,

    [ValidateSet("build", "review", "evaluate", "custom")]
    [string]$Mode = "custom",

    [string]$TaskId,

    [string]$EnvironmentId,

    [ValidateRange(1, 4)]
    [int]$Attempts,

    [string]$Branch,

    [ValidateRange(1, 4)]
    [int]$AttemptNumber,

    [switch]$Latest,

    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Script:AttemptsWasProvided = $PSBoundParameters.ContainsKey("Attempts")

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$BridgePath = Join-Path $WorkspaceRoot ".workspace_agents\codex_cloud_bridge.json"
$TaskDir = Join-Path $WorkspaceRoot ".workspace_agents\tasks"
$TemplateDir = Join-Path $WorkspaceRoot ".workspace_agents\templates"

function Assert-CodexCli {
    if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
        throw "The 'codex' CLI was not found in PATH."
    }
}

function Get-BridgeConfig {
    if (-not (Test-Path $BridgePath)) {
        return [pscustomobject]@{
            enabled = $true
            workspaceRoot = $WorkspaceRoot
            defaultEnvironmentId = ""
            defaultAttempts = 1
            defaultBranch = ""
        }
    }

    return Get-Content -Raw $BridgePath | ConvertFrom-Json
}

function Get-TemplateText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TemplateMode
    )

    if ($TemplateMode -eq "custom") {
        return ""
    }

    $templatePath = Join-Path $TemplateDir ("{0}.txt" -f $TemplateMode)
    if (-not (Test-Path $templatePath)) {
        return ""
    }

    return (Get-Content -Raw $templatePath).Trim()
}

function Build-CloudPrompt {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TemplateMode,

        [Parameter(Mandatory = $true)]
        [string]$UserPrompt
    )

    $template = Get-TemplateText -TemplateMode $TemplateMode
    if ([string]::IsNullOrWhiteSpace($template)) {
        return $UserPrompt.Trim()
    }

    return @"
$template

Tarea del usuario:
$($UserPrompt.Trim())
"@.Trim()
}

function Test-GitRepository {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        return $false
    }

    & git rev-parse --is-inside-work-tree 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Get-CurrentGitBranch {
    if (-not (Test-GitRepository)) {
        return $null
    }

    $branch = (& git branch --show-current 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $branch = ($branch | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($branch)) {
        return $null
    }

    return $branch
}

function Get-CloudTaskList {
    Assert-CodexCli
    $raw = & codex cloud list --json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list Codex Cloud tasks.`n$($raw -join "`n")"
    }

    return ($raw -join "`n") | ConvertFrom-Json
}

function Get-LatestRecordedTask {
    if (-not (Test-Path $TaskDir)) {
        return $null
    }

    $file = Get-ChildItem $TaskDir -Filter *.json -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $file) {
        return $null
    }

    return Get-Content -Raw $file.FullName | ConvertFrom-Json
}

function Resolve-TaskIdentifier {
    if (-not [string]::IsNullOrWhiteSpace($TaskId)) {
        return $TaskId
    }

    if ($Latest) {
        $record = Get-LatestRecordedTask
        if ($record -and $record.task -and -not [string]::IsNullOrWhiteSpace($record.task.id)) {
            return $record.task.id
        }

        $list = Get-CloudTaskList
        if ($list.tasks.Count -gt 0) {
            return ($list.tasks | Sort-Object updated_at -Descending | Select-Object -First 1).id
        }
    }

    throw "Provide -TaskId or use -Latest."
}

function Save-TaskRecord {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.IDictionary]$Record
    )

    if (-not (Test-Path $TaskDir)) {
        New-Item -ItemType Directory -Force $TaskDir | Out-Null
    }

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $safeMode = $Record.mode -replace "[^a-zA-Z0-9_-]", "-"
    $taskSuffix = if ($Record.task -and $Record.task.id) { $Record.task.id } else { "unknown-task" }
    $fileName = "{0}-{1}-{2}.json" -f $stamp, $safeMode, $taskSuffix
    $path = Join-Path $TaskDir $fileName

    $Record | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $path
    return $path
}

function Submit-CloudTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$UserPrompt,

        [Parameter(Mandatory = $true)]
        [string]$SubmitMode
    )

    Assert-CodexCli
    $bridge = Get-BridgeConfig

    if (-not $bridge.enabled) {
        throw "Codex Cloud bridge is disabled in .workspace_agents/codex_cloud_bridge.json."
    }

    $envIdToUse = if (-not [string]::IsNullOrWhiteSpace($EnvironmentId)) {
        $EnvironmentId
    } else {
        $bridge.defaultEnvironmentId
    }

    if ([string]::IsNullOrWhiteSpace($envIdToUse)) {
        throw "No Codex Cloud environment ID is configured. Set defaultEnvironmentId in .workspace_agents/codex_cloud_bridge.json or pass -EnvironmentId."
    }

    $attemptsToUse = if ($Script:AttemptsWasProvided) {
        $Attempts
    } elseif ($bridge.defaultAttempts) {
        [int]$bridge.defaultAttempts
    } else {
        1
    }

    $branchToUse = if (-not [string]::IsNullOrWhiteSpace($Branch)) {
        $Branch
    } elseif (-not [string]::IsNullOrWhiteSpace($bridge.defaultBranch)) {
        $bridge.defaultBranch
    } else {
        Get-CurrentGitBranch
    }

    $finalPrompt = Build-CloudPrompt -TemplateMode $SubmitMode -UserPrompt $UserPrompt
    $before = Get-CloudTaskList
    $beforeIds = @{}
    foreach ($task in $before.tasks) {
        $beforeIds[$task.id] = $true
    }

    $execArgs = @("cloud", "exec", "--env", $envIdToUse, "--attempts", "$attemptsToUse")
    if (-not [string]::IsNullOrWhiteSpace($branchToUse)) {
        $execArgs += @("--branch", $branchToUse)
    }
    $execArgs += $finalPrompt

    $execOutput = & codex @execArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Codex Cloud submission failed.`n$($execOutput -join "`n")"
    }

    Start-Sleep -Seconds 2
    $after = Get-CloudTaskList
    $newTasks = @($after.tasks | Where-Object { -not $beforeIds.ContainsKey($_.id) })
    if ($newTasks.Count -eq 0 -and $after.tasks.Count -gt 0) {
        $newTasks = @($after.tasks | Sort-Object updated_at -Descending | Select-Object -First 1)
    }

    $selectedTask = if ($newTasks.Count -gt 0) { $newTasks[0] } else { $null }

    $record = [ordered]@{
        submittedAt = (Get-Date).ToString("o")
        mode = $SubmitMode
        environmentId = $envIdToUse
        attempts = $attemptsToUse
        branch = $branchToUse
        prompt = $UserPrompt
        finalPrompt = $finalPrompt
        execOutput = ($execOutput -join "`n")
        task = $selectedTask
    }

    $path = Save-TaskRecord -Record $record

    [pscustomobject]@{
        path = $path
        task = $selectedTask
        execOutput = ($execOutput -join "`n")
        mode = $SubmitMode
        environmentId = $envIdToUse
        attempts = $attemptsToUse
        branch = $branchToUse
    }
}

switch ($Command) {
    "submit" {
        if ([string]::IsNullOrWhiteSpace($Prompt)) {
            throw "submit requires -Prompt."
        }

        $result = Submit-CloudTask -UserPrompt $Prompt -SubmitMode $Mode
        Write-Output ("Submitted mode: {0}" -f $result.mode)
        Write-Output ("Environment: {0}" -f $result.environmentId)
        if (-not [string]::IsNullOrWhiteSpace($result.branch)) {
            Write-Output ("Branch: {0}" -f $result.branch)
        }
        Write-Output ("Attempts: {0}" -f $result.attempts)
        Write-Output ("Record: {0}" -f $result.path)
        if ($result.task) {
            Write-Output ("Task ID: {0}" -f $result.task.id)
            if ($result.task.url) {
                Write-Output ("Task URL: {0}" -f $result.task.url)
            }
            if ($result.task.status) {
                Write-Output ("Status: {0}" -f $result.task.status)
            }
        } else {
            Write-Output "Task submitted, but the new task could not be identified from the task list yet."
        }
        break
    }

    "status" {
        $resolvedTaskId = Resolve-TaskIdentifier
        & codex cloud status $resolvedTaskId
        break
    }

    "diff" {
        $resolvedTaskId = Resolve-TaskIdentifier
        $diffArgs = @("cloud", "diff", $resolvedTaskId)
        if ($PSBoundParameters.ContainsKey("AttemptNumber")) {
            $diffArgs += @("--attempt", "$AttemptNumber")
        }
        & codex @diffArgs
        break
    }

    "apply" {
        $resolvedTaskId = Resolve-TaskIdentifier
        $applyArgs = @("cloud", "apply", $resolvedTaskId)
        if ($PSBoundParameters.ContainsKey("AttemptNumber")) {
            $applyArgs += @("--attempt", "$AttemptNumber")
        }
        & codex @applyArgs
        break
    }

    "list" {
        $list = Get-CloudTaskList
        if ($Json) {
            $list | ConvertTo-Json -Depth 20
        } else {
            if ($list.tasks.Count -eq 0) {
                Write-Output "No Codex Cloud tasks found."
            } else {
                $list.tasks |
                    Sort-Object updated_at -Descending |
                    Select-Object id, status, title, updated_at, environment_id, url |
                    Format-Table -AutoSize
            }
        }
        break
    }

    "config" {
        $config = Get-BridgeConfig
        $config | ConvertTo-Json -Depth 10
        break
    }
}
