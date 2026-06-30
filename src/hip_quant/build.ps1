param(
    [string]$Output = "hip_quantize.dll",
    [string]$Arch = $env:HIP_QUANT_ARCH,
    [string]$RocmBin = $env:HIP_QUANT_ROCM_BIN
)

$ErrorActionPreference = "Stop"

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

if ([string]::IsNullOrWhiteSpace($Arch)) {
    $Arch = "all"
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
$src_dir_arg = Get-ShortPath $src_dir
$out_file = Join-Path $src_dir $Output
$out_file_arg = Join-Path $src_dir_arg $Output

Write-Host "Compiling HIP quantization DLL..."
Write-Host "hipcc: $hipcc"
Write-Host "arch: $Arch"
Write-Host "source: $src_dir"
Write-Host "output: $out_file"

if ($Arch -eq "all") {
    $archs = @("gfx90a", "gfx942", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1200", "gfx1201")
}
else {
    $archs = $Arch.Split(',')
}

$offload_args = @()
foreach ($a in $archs) {
    $offload_args += "--offload-arch=$a"
}

$arg_list = @(
    "-O3",
    "-mno-wavefrontsize64",
    "-ffp-contract=off",
    "-shared",
    "-I", $src_dir_arg
)

if ($archs -match 'gfx9') {
    $arg_list += "-DHIP_QUANT_HAS_CDNA=1"
}

$arg_list += $offload_args
$arg_list += @("-o", $out_file_arg, (Join-Path $src_dir_arg "hip_quantize.cpp"))

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
Write-Host "Architectures: $($archs -join ', ')"
