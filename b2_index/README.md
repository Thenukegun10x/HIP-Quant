# B2 wheel index (custom PyTorch 2.15)

`torch_index.html` is the `--find-links` index page for the custom PyTorch
2.15 / ROCm 7.14 wheels. It is not used at runtime — it is uploaded to the
public Backblaze B2 bucket so `pip` can resolve `torch==2.15.0a0+gitf07882e`.

## Why not a PEP 503 `/simple/` index?
B2 has no directory-index behaviour and rclone cannot create object keys that
end in `/`, so `GET /simple/torch/` cannot be served from B2 alone. A single
static HTML page with `--find-links` works today. (A Cloudflare rewrite from
`/simple/(.*)/` to `/file/Torchs/simple/$1/index.html` would enable a real
`/simple/` index if wanted.)

## Upload
```powershell
rclone copyto b2_index/torch_index.html torch-upload:Torchs/torch_index.html
```

## Install
```powershell
pip install "hip-quant==2.2.1.post215" `
  --find-links https://dl.hipquant.download/file/Torchs/torch_index.html
```

## Gotchas
- **Filenames must match wheel metadata.** The objects are
  `torch-2.15.0a0+gitf07882e-…`, `torchvision-0.30.0a0+ac8d215-…`,
  `torchaudio-2.11.0a0+b85c99c-…`. A name like `torch-2.15.0-…` is rejected by
  pip ("inconsistent version").
- **`+` must be percent-encoded as `%2B`** in URLs (`…a0%2Bgitf…`); a literal
  `+` in the path returns 404.
- The bucket is on B2 cluster **f005**; the Cloudflare domain preserves the
  full path, so the canonical URL is
  `https://dl.hipquant.download/file/Torchs/<key>` (or
  `https://f005.backblazeb2.com/file/Torchs/<key>`).

## Updating a wheel
```powershell
rclone copyto <new>.whl "torch-upload:Torchs/<name>-<version>-cp312-cp312-win_amd64.whl"
rclone deletefile "torch-upload:Torchs/<old>.whl" --b2-hard-delete
```
Then edit `torch_index.html` if the filename changed and re-upload it.
