param(
    [string]$Output = "hip_quantize.dll"
)

$ErrorActionPreference = "Stop"

$hipcc = "C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe"

if (!(Test-Path $hipcc)) {
    Write-Error "hipcc not found at $hipcc"
    exit 1
}

$src_dir = "C:\Users\armor\Desktop\MEDUSA framwork\src\hip_quant"
$out_file = "C:\Users\armor\Desktop\MEDUSA framwork\src\$Output"

Write-Host "Compiling HIP quantization DLL..."

$arg_list = @(
    "-O3",
    "-ffp-contract=off",
    "-shared",
    "-I", $src_dir,
    "--offload-arch=gfx1201",
    "-o", "$src_dir\hip_quantize.dll",
    "$src_dir\hip_quantize.cpp"
)

$result = & $hipcc @arg_list 2>&1
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    Write-Error "Compilation failed with exit code $exit"
    Write-Host $result
    exit $exit
}

Write-Host $result
Write-Host "DLL created: $out_file"
