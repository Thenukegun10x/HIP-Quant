param(
    [string]$Output = "hip_quantize.dll",
    [switch]$CDNA,
    [switch]$All,
    [string]$Arch = "",
    [string]$RocmBin = $env:HIP_QUANT_ROCM_BIN
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
    $RocmBin = "C:\Program Files\AMD\ROCm\7.1\bin"
}

$hipcc = Join-Path $RocmBin "hipcc.exe"

if (!(Test-Path $hipcc)) {
    Write-Error "hipcc not found at $hipcc"
    exit 1
}

$src_dir = "C:\Users\armor\Desktop\MEDUSA framwork\src\hip_quant"
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
