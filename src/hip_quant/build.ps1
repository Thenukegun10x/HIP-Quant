param(
    [string]$Output = "hip_quantize.dll",
    [string]$Arch = $env:HIP_QUANT_ARCH,
    [string]$RocmBin = $env:HIP_QUANT_ROCM_BIN
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Arch)) {
    $Arch = "gfx1201"
}
if ([string]::IsNullOrWhiteSpace($RocmBin)) {
    $RocmBin = "C:\Program Files\AMD\ROCm\7.1\bin"
}

$hipcc = Join-Path $RocmBin "hipcc.exe"
if (!(Test-Path -LiteralPath $hipcc)) {
    Write-Error "hipcc not found at $hipcc"
    exit 1
}

$src_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$out_file = Join-Path $src_dir $Output

Write-Host "Compiling HIP quantization DLL..."
Write-Host "hipcc: $hipcc"
Write-Host "arch: $Arch"
Write-Host "source: $src_dir"
Write-Host "output: $out_file"

$arg_list = @(
    "-O3",
    "-ffp-contract=off",
    "-shared",
    "-I", $src_dir,
    "--offload-arch=$Arch",
    "-o", $out_file,
    (Join-Path $src_dir "hip_quantize.cpp")
)

$old_eap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$result = & $hipcc @arg_list 2>&1
$exit = $LASTEXITCODE
$ErrorActionPreference = $old_eap

if ($exit -ne 0) {
    Write-Host $result
    Write-Error "Compilation failed with exit code $exit"
    exit $exit
}

Write-Host $result
Write-Host "DLL created: $out_file"
