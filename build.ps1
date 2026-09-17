param(
    [string]$Output = "hip_quantize.dll",
    [switch]$CDNA,
    [switch]$All,
    [string]$Arch = "",
    [string]$RocmBin = $env:HIP_QUANT_ROCM_BIN,
    [switch]$SkipSmi = $false,
    [switch]$SmiOnly = $false
)

$ErrorActionPreference = "Continue"

function Get-ShortPath([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        return $Path
    }
    try {
        Add-Type -MemberDefinition '[DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern int GetShortPathName(string longPath, System.Text.StringBuilder shortPath, int shortPathLength);' -Name Win32ShortPath -Namespace Native -ErrorAction SilentlyContinue | Out-Null
        $buffer = New-Object System.Text.StringBuilder 260
        $result = [Native.Win32ShortPath]::GetShortPathName($Path, $buffer, $buffer.Capacity)
        if ($result -gt 0) {
            return $buffer.ToString()
        }
    }
    catch {
    }
    return $Path
}

if ([string]::IsNullOrWhiteSpace($RocmBin)) {
    if ($env:HIP_QUANT_ROCM_BIN -and (Test-Path $env:HIP_QUANT_ROCM_BIN)) {
        $RocmBin = $env:HIP_QUANT_ROCM_BIN
    } elseif ($env:ROCM_PATH -and (Test-Path (Join-Path $env:ROCM_PATH "bin"))) {
        $RocmBin = Join-Path $env:ROCM_PATH "bin"
    } elseif ($env:HIP_PATH -and (Test-Path (Join-Path $env:HIP_PATH "bin"))) {
        $RocmBin = Join-Path $env:HIP_PATH "bin"
    } else {
        $rocmCandidates = Get-ChildItem -Path "C:\Program Files\AMD\ROCm" -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "bin" } |
            Where-Object { Test-Path (Join-Path $_ "hipcc.exe") }
        if ($rocmCandidates) {
            $RocmBin = $rocmCandidates[0]
        } else {
            $RocmBin = "C:\Program Files\AMD\ROCm\7.1\bin"
        }
    }
}

$hipcc = Join-Path $RocmBin "hipcc.exe"

if (!(Test-Path $hipcc)) {
    Write-Error "hipcc not found at $hipcc"
    exit 1
}

$src_dir = $PSScriptRoot
$src_dir_arg = Get-ShortPath $src_dir
$out_file = Join-Path $src_dir $Output
$out_file_arg = Join-Path $src_dir_arg $Output

Write-Host "Compiling HIP quantization DLL..."

# Determine target architectures
$archs = @()

if ($All) {
    # All supported architectures
    $archs = @(
        "gfx90a",      # CDNA 2 (MI250)
        "gfx942",      # CDNA 3 (MI300X)
        "gfx1100",     # RDNA 3
        "gfx1101",     # RDNA 3
        "gfx1102",     # RDNA 3
        "gfx1103",     # RDNA 3
        "gfx1200",     # RDNA 4
        "gfx1201"      # RDNA 4
    )
}
elseif ($CDNA) {
    # RDNA4 + CDNA targets
    $archs = @(
        "gfx90a",      # CDNA 2
        "gfx942",      # CDNA 3
        "gfx1200",     # RDNA 4
        "gfx1201"      # RDNA 4
    )
    Write-Host "Building with CDNA + RDNA4 support"
}
elseif ($Arch -ne "") {
    # User-specified arch
    $archs = $Arch.Split(',')
}
else {
    # Default: one DLL for all supported CDNA/RDNA targets.
    $archs = @(
        "gfx90a",      # CDNA 2 (MI250)
        "gfx942",      # CDNA 3 (MI300X)
        "gfx1100",     # RDNA 3
        "gfx1101",     # RDNA 3
        "gfx1102",     # RDNA 3
        "gfx1103",     # RDNA 3
        "gfx1200",     # RDNA 4
        "gfx1201"      # RDNA 4
    )
}

$offload_args = @()
foreach ($a in $archs) {
    $offload_args += "--offload-arch=$a"
}

Write-Host "Target architectures: $($archs -join ', ')"

$arg_list = @(
    "-O3",
    "-mno-wavefrontsize64",
    "-ffp-contract=off",
    "-shared",
    "-Wno-ignored-attributes",
    "-D_CRT_SECURE_NO_WARNINGS",
    "-I", $src_dir_arg
)

# Add arch-specific defines
if ($archs -match 'gfx9') {
    $arg_list += "-DHIP_QUANT_HAS_CDNA=1"
}

# Add offload arch flags
$arg_list += $offload_args
$arg_list += @("-o", $out_file_arg, (Join-Path $src_dir_arg "hip_quantize.cpp"))

$result = & $hipcc @arg_list 2>&1
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    Write-Error "Compilation failed with exit code $exit"
    Write-Host $result
    exit $exit
}

Write-Host $result
Write-Host "DLL created: $out_file"
Write-Host "Architectures: $($archs -join ', ')"

# ── gpu-smi bundling ─────────────────────────────────────────────────────
# gpu-smi is its own project (github.com/Thenukegun10x/GPU-SMI). Fetch the
# current release asset (checksum-verified) into tools/ on every build; it is
# bundled in the wheel via package-data. Falls back to a local Rust build, then
# to shipping without it. Set -SkipSmi to skip.
if ($SmiOnly) { $SkipSmi = $false }
if (-not $SkipSmi) {
    $toolsDir = Join-Path $src_dir "tools"
    if (!(Test-Path $toolsDir)) { New-Item -ItemType Directory -Path $toolsDir | Out-Null }
    $fetchScript = Join-Path $toolsDir "fetch_gpu_smi.py"

    $py = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)
    if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source) }

    $haveSmi = $false
    if ($py -and (Test-Path $fetchScript)) {
        Write-Host "`nFetching gpu-smi from GitHub Releases..."
        & $py $fetchScript
        $haveSmi = ($LASTEXITCODE -eq 0)
    }

    if (-not $haveSmi) {
        $smiManifest = Join-Path $src_dir "gpu-smi-src\Cargo.toml"
        $smiSrc = Join-Path $src_dir "gpu-smi-src"
        if (Test-Path $smiManifest) {
            Write-Host "`nRelease fetch unavailable — building gpu-smi from gpu-smi-src (cargo)..."
            $cargoBin = @(
                (Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"),
                (Get-Command cargo -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
            ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique -First 1
            if (-not $cargoBin) { $cargoBin = "cargo" }
            if ($cargoBin -is [array]) { $cargoBin = $cargoBin[0] }
            $smiOut = Join-Path $toolsDir "gpu-smi.exe"
            & $cargoBin build --release --manifest-path $smiManifest 2>&1 | Write-Host
            if ($LASTEXITCODE -eq 0) {
                $built = Join-Path $smiSrc "target\release\gpu-smi.exe"
                if (Test-Path $built) {
                    Copy-Item -Force $built $smiOut
                    Write-Host "gpu-smi vendored: $smiOut ($( (Get-Item $smiOut).Length / 1KB ) KB)"
                } else {
                    Write-Warning "cargo built but $built not found"
                }
            } else {
                Write-Warning "cargo build failed (exit $LASTEXITCODE)"
            }
        } else {
            Write-Warning "gpu-smi could not be fetched and gpu-smi-src is absent — wheel ships without a bundled gpu-smi"
        }
    }
}
if ($SmiOnly) { exit 0 }
