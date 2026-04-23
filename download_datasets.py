"""
download_datasets.py
Downloads SummEval and Frank datasets manually
(since the summac library's auto-download is broken)
"""
import os
import json
import requests

BENCHMARK = "./summac_benchmark"

def download_summeval():
    """Download SummEval from Google Drive"""
    folder = os.path.join(BENCHMARK, "summeval")
    target = os.path.join(folder, "model_annotations.aligned.scored.jsonl")
    
    if os.path.exists(target) and os.path.getsize(target) > 1000:
        print(f"SummEval already exists ({os.path.getsize(target)} bytes), skipping.")
        return True
    
    os.makedirs(folder, exist_ok=True)
    
    file_id = "1d2Iaz3jNraURP1i7CfTqPIj8REZMJ3tS"
    
    # Try direct Google Drive download
    print("Downloading SummEval from Google Drive...")
    
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    response = session.get(url, stream=True)
    
    # Check if we got a confirmation page (large file warning)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break
    
    # Also try confirm=t approach
    if response.headers.get("Content-Type", "").startswith("text/html"):
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        response = session.get(url, stream=True)
    
    with open(target, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    size = os.path.getsize(target)
    print(f"  Downloaded: {size} bytes")
    
    # Validate it's valid JSONL
    try:
        with open(target, "r") as f:
            first_line = f.readline()
            json.loads(first_line)
        print("  Validated: valid JSONL")
        return True
    except Exception as e:
        print(f"  WARNING: File may be corrupted: {e}")
        print(f"  First 200 chars: {open(target, 'r').read(200)}")
        return False


def download_frank():
    """Download Frank from GitHub"""
    folder = os.path.join(BENCHMARK, "frank")
    
    fns = ["human_annotations_sentence.json", "validation_split.txt", "test_split.txt"]
    all_exist = all(
        os.path.exists(os.path.join(folder, fn)) and os.path.getsize(os.path.join(folder, fn)) > 100
        for fn in fns
    )
    
    if all_exist:
        print("Frank already exists, skipping.")
        return True
    
    os.makedirs(folder, exist_ok=True)
    
    print("Downloading Frank from GitHub...")
    base_url = "https://raw.githubusercontent.com/artidoro/frank/main/data"
    
    for fn in fns:
        url = f"{base_url}/{fn}"
        print(f"  Downloading {fn}...")
        r = requests.get(url)
        if r.status_code != 200:
            print(f"  ERROR: Failed to download {fn} (HTTP {r.status_code})")
            return False
        
        with open(os.path.join(folder, fn), "w") as f:
            f.write(r.text)
        print(f"  OK ({len(r.text)} bytes)")
    
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Downloading SummaC Benchmark datasets")
    print("=" * 50)
    
    s1 = download_summeval()
    print()
    s2 = download_frank()
    
    print()
    print("=" * 50)
    print(f"SummEval: {'OK' if s1 else 'FAILED'}")
    print(f"Frank:    {'OK' if s2 else 'FAILED'}")
    print("=" * 50)
    
    if s1 and s2:
        print("\nAll downloads complete. You can now run:")
        print("  .\\venv\\Scripts\\python.exe new_script.py")
    else:
        print("\nSome downloads failed. Check errors above.")
