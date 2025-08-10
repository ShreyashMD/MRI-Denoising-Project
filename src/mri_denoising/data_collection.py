import os
import tempfile
import requests
import numpy as np
import nibabel as nib
from typing import Optional, List


class OpenNeuroStreamer:
    """Stream fMRI .nii.gz files from the OpenNeuro S3 bucket."""

    def __init__(self, dataset_id: str = "ds002306") -> None:
        self.dataset_id = dataset_id
        self.base_url = f"https://s3.amazonaws.com/openneuro.org/{dataset_id}"

    def check_url_exists(self, url: str) -> bool:
        """Check if a file exists on the remote server."""
        try:
            response = requests.head(url, timeout=15)
            return response.status_code == 200
        except Exception:
            return False

    def stream_nii_file(self, url: str) -> Optional[np.ndarray]:
        """Stream a NIfTI file directly into memory as a numpy array."""
        print(f"Streaming: {os.path.basename(url)}")
        temp_path = None
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                temp_path = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            img = nib.load(temp_path)
            return img.get_fdata(dtype=np.float32)
        except Exception as e:
            print(f"Error streaming file: {e}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def load_files_for_chunk(
        self,
        subjects: List[str],
        num_files: int,
        downloaded_urls: set,
    ) -> (List[np.ndarray], set):
        """Download up to `num_files` fMRI runs from the provided subjects."""
        data = []
        for s in subjects:
            if len(data) >= num_files:
                break
            for r in range(1, 6):
                if len(data) >= num_files:
                    break
                url = f"{self.base_url}/{s}/func/{s}_task-training_run-{r:02d}_bold.nii.gz"
                if url in downloaded_urls:
                    continue
                if self.check_url_exists(url):
                    d = self.stream_nii_file(url)
                    if d is not None:
                        data.append(d)
                        downloaded_urls.add(url)
        return data, downloaded_urls


def main():
    """Example usage for downloading a small dataset chunk."""
    import argparse

    parser = argparse.ArgumentParser(description="Download fMRI data chunk from OpenNeuro")
    parser.add_argument("--output", default="fmri_dataset_chunk.npy", help="Output .npy file")
    parser.add_argument("--num-files", type=int, default=20, help="Number of runs to download")
    args = parser.parse_args()

    print("NOTEBOOK 1A: Downloading Chunk")
    streamer = OpenNeuroStreamer()
    subjects_pool = [f"sub-{i:02d}" for i in range(1, 11)]

    chunk_data, _ = streamer.load_files_for_chunk(subjects_pool, args.num_files, set())
    if chunk_data:
        np.save(args.output, np.array(chunk_data, dtype=np.float32))
        print(f"\nChunk saved to {args.output}")
    else:
        print("No data downloaded.")


if __name__ == "__main__":
    main()
