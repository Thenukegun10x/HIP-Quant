import urllib.request
import re
import json
import html as html_lib
from collections import Counter

url = "https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    page = resp.read().decode("utf-8")

matches = re.findall(r'data-target="BlobGgufViewer" data-props="([^"]+)"', page)
if not matches:
    matches = re.findall(r'data-props="([^"]+)"', page)

found = False
for m in matches:
    raw = html_lib.unescape(m)
    if "tensors" in raw:
        data = json.loads(raw)
        tensors = data.get("tensors", [])
        if not tensors and "gguf" in data:
            tensors = data["gguf"].get("tensors", [])
        if tensors:
            found = True
            print(f"Successfully retrieved {len(tensors)} tensors from Hugging Face!")
            type_counts = Counter(t.get("type", t.get("dtype")) for t in tensors)
            print("\n=== Exact Quantization Types in Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf ===")
            for t_type, count in type_counts.most_common():
                print(f"  {str(t_type):<15}: {count} tensors")
            
            print("\n=== Sample Tensors by Type ===")
            seen = set()
            for t in tensors:
                ttype = t.get("type", t.get("dtype"))
                name = t.get("name")
                shape = t.get("shape")
                if ttype not in seen:
                    print(f"  [{ttype:<10}] {name:<40} shape={shape}")
                    seen.add(ttype)
            break

if not found:
    print("Could not find BlobGgufViewer props in HTML, searching for alternative JSON endpoints...")
    # Check if there is an api endpoint
    m_endpoints = re.findall(r'/api/models/[^"\']+', page)
    print("Found API endpoints:", set(m_endpoints))
