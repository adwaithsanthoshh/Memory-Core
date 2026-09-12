import os
import urllib.request
import sys

def download_with_progress():
    url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
    target_dir = "data/core50_128x128"
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "core50_128x128.zip")
    
    print(f"Starting robust download of CORe50 to {zip_path}...")
    
    def report(count, block_size, total_size):
        downloaded = count * block_size
        percent = int(downloaded * 100 / total_size)
        # Only print every 5% to avoid log spam, and force flush
        if count % 1000 == 0 or downloaded >= total_size:
            print(f"Downloaded: {percent}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", flush=True)

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=report)
        print("Download finished successfully!", flush=True)
    except Exception as e:
        print(f"Download failed: {e}", flush=True)

if __name__ == "__main__":
    download_with_progress()
