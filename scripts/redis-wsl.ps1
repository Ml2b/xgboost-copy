param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "stop", "restart", "status", "ping")]
    [string]$Action
)

$serviceName = "redis-server"
$distro = "Ubuntu"

function Invoke-WslRedis {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [switch]$AsRoot
    )

    $args = @("-d", $distro)
    if ($AsRoot) {
        $args += @("-u", "root")
    }
    $args += "--", "bash", "-lc", $Command
    & wsl @args
    $script:WslExitCode = $LASTEXITCODE
}

switch ($Action) {
    "start" {
        Invoke-WslRedis -AsRoot -Command "systemctl start $serviceName && systemctl is-active $serviceName"
        exit $script:WslExitCode
    }
    "stop" {
        Invoke-WslRedis -AsRoot -Command "systemctl stop $serviceName && systemctl is-active $serviceName || true"
        exit $script:WslExitCode
    }
    "restart" {
        Invoke-WslRedis -AsRoot -Command "systemctl restart $serviceName && systemctl is-active $serviceName"
        exit $script:WslExitCode
    }
    "status" {
        Invoke-WslRedis -AsRoot -Command "systemctl is-active $serviceName && systemctl status $serviceName --no-pager --lines=3"
        exit $script:WslExitCode
    }
    "ping" {
        Invoke-WslRedis -Command "redis-cli ping"
        exit $script:WslExitCode
    }
}
