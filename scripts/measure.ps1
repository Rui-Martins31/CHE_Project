<#
.SYNOPSIS
  Run a harknn binary several times and measure time, energy, power and clock frequency.

.DESCRIPTION
  Energy comes from the Windows "Energy Meter" performance counters, which expose the
  Intel RAPL counters (no admin rights or extra drivers needed):
    RAPL_Package0_PKG  whole CPU package    (counter unit: picowatt-hour)
    RAPL_Package0_PP0  CPU cores only
  The counters refresh about every 8 ms, so the script polls them during each run and keeps
  a timeline. Since the kNN loop is the last phase of the program, the kNN-only energy is the
  energy in the window [end - knn_time, end], where knn_time is printed by the program.

  Frequency is the "Actual Frequency" counter of the logical CPU the program is pinned to,
  averaged over the whole run and over the kNN window.

  The program is pinned to one P-core (-Cpu, default 12) and this script pins itself to an
  E-core (-MonitorCpu, default 2), so the polling does not steal time from the program.

  Writes one CSV row per repetition (warm-up runs are not written).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\measure.ps1 -Exe knn\bin\harknn.exe -Reps 10 -OutCsv out.csv
#>
param(
  [Parameter(Mandatory)] [string] $Exe,
  [int] $Reps = 10,
  [int] $Warmup = 1,
  [int] $Cpu = 12,          # logical CPU for the program (P-cores on the Core Ultra 7 265H: 0,1,10-13)
  [int] $MonitorCpu = 2,    # logical CPU for this script (an E-core)
  [int] $PollMs = 2,
  [Parameter(Mandatory)] [string] $OutCsv,
  [string] $WorkDir = (Split-Path -Parent $Exe)
)
$ErrorActionPreference = 'Stop'
$PWH_TO_J = 3.6e-9

# 1 ms timer resolution, otherwise sleeps round up to ~16 ms (Start-Sleep always does)
Add-Type -Namespace Win32 -Name WinMM -MemberDefinition '[DllImport("winmm.dll")] public static extern uint timeBeginPeriod(uint p);'
[void][Win32.WinMM]::timeBeginPeriod(1)

$self = [Diagnostics.Process]::GetCurrentProcess()
$self.ProcessorAffinity = [IntPtr](1 -shl $MonitorCpu)
$self.PriorityClass = 'High'

$pkg   = New-Object Diagnostics.PerformanceCounter('Energy Meter', 'Energy', 'RAPL_Package0_PKG')
$cores = New-Object Diagnostics.PerformanceCounter('Energy Meter', 'Energy', 'RAPL_Package0_PP0')
$freq  = New-Object Diagnostics.PerformanceCounter('Processor Information', 'Actual Frequency', "0,$Cpu")
[void]$freq.NextSample()

function Measure-Once {
  $tl = New-Object Collections.Generic.List[object]   # timeline: (t, pkg, cores, freq sample)
  $p = New-Object Diagnostics.Process
  $p.StartInfo.FileName = (Resolve-Path $Exe).Path
  $p.StartInfo.WorkingDirectory = $WorkDir
  $p.StartInfo.UseShellExecute = $false
  $p.StartInfo.RedirectStandardOutput = $true
  $p.StartInfo.RedirectStandardError = $true

  $sw = [Diagnostics.Stopwatch]::StartNew()
  $tl.Add(@(0.0, $pkg.RawValue, $cores.RawValue, $freq.NextSample()))
  [void]$p.Start()
  try { $p.ProcessorAffinity = [IntPtr](1 -shl $Cpu); $p.PriorityClass = 'High' } catch { }
  $outTask = $p.StandardOutput.ReadToEndAsync()
  $errTask = $p.StandardError.ReadToEndAsync()
  while (-not $p.HasExited) {
    $tl.Add(@($sw.Elapsed.TotalSeconds, $pkg.RawValue, $cores.RawValue, $freq.NextSample()))
    [Threading.Thread]::Sleep($PollMs)
  }
  $p.WaitForExit()
  $wall = $sw.Elapsed.TotalSeconds
  # one more sample after the counter has refreshed, so the last ~8 ms are included
  Start-Sleep -Milliseconds 10
  $tl.Add(@($sw.Elapsed.TotalSeconds, $pkg.RawValue, $cores.RawValue, $freq.NextSample()))
  $out = $outTask.Result + $errTask.Result

  $knn = [double]::NaN; $acc = [double]::NaN
  if ($out -match 'Total execution time:\s+([0-9.]+) s') { $knn = [double]$Matches[1] }
  if ($out -match 'accuracy = ([0-9.]+)') { $acc = [double]$Matches[1] }

  $first = $tl[0]; $last = $tl[$tl.Count - 1]
  $pkgJ = ($last[1] - $first[1]) * $PWH_TO_J
  $coresJ = ($last[2] - $first[2]) * $PWH_TO_J
  $f = [Diagnostics.CounterSample]::Calculate($first[3], $last[3])

  # kNN window: from (wall - knn) to the end
  $kPkgJ = [double]::NaN; $kCoresJ = [double]::NaN; $kF = [double]::NaN
  if (-not [double]::IsNaN($knn)) {
    $tStart = $wall - $knn
    $s = $first
    foreach ($row in $tl) { if ($row[0] -le $tStart) { $s = $row } else { break } }
    $kPkgJ = ($last[1] - $s[1]) * $PWH_TO_J
    $kCoresJ = ($last[2] - $s[2]) * $PWH_TO_J
    $kF = [Diagnostics.CounterSample]::Calculate($s[3], $last[3])
    # the window starts at sample s, not exactly at tStart: scale to the kNN time
    $span = $last[0] - $s[0]
    if ($span -gt 0) { $kPkgJ *= $knn / $span; $kCoresJ *= $knn / $span }
  }
  [pscustomobject]@{
    exit_code = $p.ExitCode; wall_s = $wall; knn_s = $knn; accuracy = $acc
    pkg_J = $pkgJ; cores_J = $coresJ; pkg_W = $pkgJ / $wall; freq_MHz = $f
    knn_pkg_J = $kPkgJ; knn_cores_J = $kCoresJ; knn_pkg_W = $kPkgJ / $knn; knn_freq_MHz = $kF
    samples = $tl.Count
  }
}

for ($i = 0; $i -lt $Warmup; $i++) { [void](Measure-Once) }

$rows = for ($r = 1; $r -le $Reps; $r++) {
  $m = Measure-Once
  $m | Add-Member -NotePropertyName rep -NotePropertyValue $r -PassThru
  if ($m.exit_code -ne 0) { break }   # crashed binary: no point repeating
}
$inv = [Globalization.CultureInfo]::InvariantCulture
$cols = 'rep','exit_code','wall_s','knn_s','accuracy','pkg_J','cores_J','pkg_W','freq_MHz','knn_pkg_J','knn_cores_J','knn_pkg_W','knn_freq_MHz','samples'
$lines = @($cols -join ',') + ($rows | ForEach-Object {
  $row = $_; ($cols | ForEach-Object { ([double]$row.$_).ToString('G6', $inv) }) -join ','
})
[IO.File]::WriteAllLines($OutCsv, [string[]]$lines)
