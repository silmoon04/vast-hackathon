<#
.SYNOPSIS
  Open the laptop -> lab VM tunnels in their own window (Windows OpenSSH has no -f).

.EXAMPLE
  .\scripts\tunnel.ps1 -Alias bc-team-7
  .\scripts\tunnel.ps1 -Alias bc-team-7 -Forward "18001:10.0.0.5:8001","18003:10.0.0.5:8003"

.NOTES
  - SOCKS (-D) is for the private VSS/Ingress names: clients use socks5h://127.0.0.1:<port> (remote DNS).
    Apply it per client (VAST_SOCKS), never as a global HTTPS_PROXY.
  - Do NOT -L forward INGRESS_URL: nginx routes on Host/SNI. -L is only for raw IP:port GPU endpoints.
  - VastDB SQL needs the Linux ADBC driver: run those queries on the VM, not through this tunnel.
#>
param(
  [Parameter(Mandatory = $true)][string]$Alias,
  [int]$Socks = 1080,
  [string[]]$Forward = @(),
  [switch]$Foreground
)

$sshArgs = @('-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30', '-D', "127.0.0.1:$Socks")
foreach ($f in $Forward) { $sshArgs += @('-L', "127.0.0.1:$f") }
$sshArgs += $Alias

$busy = Get-NetTCPConnection -LocalPort $Socks -State Listen -ErrorAction SilentlyContinue
if ($busy) { Write-Warning "port $Socks already listening (pid $($busy.OwningProcess)); an old tunnel may be running" }

if ($Foreground) {
  & ssh @sshArgs
} else {
  $p = Start-Process -FilePath 'ssh' -ArgumentList $sshArgs -PassThru -WindowStyle Minimized
  Write-Host "tunnel pid $($p.Id) (close its window or Stop-Process -Id $($p.Id) to end it)"
}
Write-Host "In client shells:  `$env:VAST_SOCKS = 'socks5h://127.0.0.1:$Socks'"
foreach ($f in $Forward) {
  $local = $f.Split(':')[0]
  Write-Host "  forwarded GPU port -> http://127.0.0.1:$local"
}
