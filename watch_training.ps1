# Live training viewer.
#
# Training runs detached, so its tqdm bar goes into a log file instead of a
# terminal. tqdm redraws by emitting carriage returns, and `Get-Content -Wait`
# appends each redraw as a new line -- which is why tailing the log scrolls
# instead of animating.
#
# This reads the same file and rebuilds the original view: completed lines
# scroll normally, the newest progress fragment is redrawn in place.
#
#   powershell -ExecutionPolicy Bypass -File watch_training.ps1
#
# Ctrl+C closes the viewer. It never touches the training process -- this only
# ever opens the log for reading.

param(
    [string]$Log = "D:\Eye-detection\train_5disease.log",
    [int]$IntervalMs = 400
)

$ErrorActionPreference = "Stop"

# lines worth scrolling; everything else is progress-bar churn
$keep = '^(Epoch \d+ \|| *-> saved new best|Early stopping|Done\.|Training on:|Supervised heads:|NOT supervised|Resuming from)'

$printed = New-Object 'System.Collections.Generic.HashSet[string]'
$lastBar = ""
$announced = $false

Write-Host "watching $Log  (Ctrl+C to close -- training keeps running)" -ForegroundColor DarkGray
Write-Host ""

while ($true) {
    if (-not (Test-Path $Log)) {
        Start-Sleep -Milliseconds 800
        continue
    }

    # FileShare ReadWrite: never block the trainer's own writes
    try {
        $fs = [System.IO.File]::Open($Log, 'Open', 'Read', 'ReadWrite')
        $sr = New-Object System.IO.StreamReader($fs)
        $text = $sr.ReadToEnd()
        $sr.Close(); $fs.Close()
    } catch {
        Start-Sleep -Milliseconds $IntervalMs
        continue
    }

    $lines = $text -split "`n"

    # --- completed lines: print each one once, in order ---
    for ($i = 0; $i -lt $lines.Count - 1; $i++) {
        # a line may carry earlier bar redraws before it; keep the final piece
        $line = ($lines[$i] -split "`r")[-1].TrimEnd()
        if ($line -eq "" -or -not ($line -match $keep)) { continue }
        $key = "$i|$line"
        if ($printed.Add($key)) {
            if ($lastBar -ne "") {
                Write-Host ("`r" + (" " * ($lastBar.Length + 2)) + "`r") -NoNewline
                $lastBar = ""
            }
            $colour = 'Gray'
            if ($line -match 'saved new best') { $colour = 'Green' }
            elseif ($line -match '^Epoch')     { $colour = 'Cyan' }
            elseif ($line -match 'Early stopping|Done\.') { $colour = 'Yellow' }
            elseif ($line -match 'NOT supervised')        { $colour = 'DarkYellow' }
            Write-Host $line -ForegroundColor $colour
        }
    }

    # --- the live bar: the newest fragment after the last newline ---
    $tailPart = $lines[-1]
    $frags = @($tailPart -split "`r" | Where-Object { $_.Trim() -ne "" })
    if ($frags.Count -gt 0) {
        $bar = $frags[-1].TrimEnd()
        if ($bar -ne $lastBar) {
            $pad = [Math]::Max(0, $lastBar.Length - $bar.Length)
            Write-Host ("`r" + $bar + (" " * $pad)) -NoNewline -ForegroundColor White
            $lastBar = $bar
        }
    }

    if (-not $announced -and $text -match 'Done\.') {
        Write-Host ""
        Write-Host "training finished" -ForegroundColor Yellow
        $announced = $true
    }

    Start-Sleep -Milliseconds $IntervalMs
}
