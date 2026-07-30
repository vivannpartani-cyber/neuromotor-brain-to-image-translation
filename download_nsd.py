import os
import requests
import tarfile
import urllib.request
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

NSD_S3_BASE = "https://natural-scenes-dataset.s3.amazonaws.com"
DATA_DIR = "data/nsd"
COCO_IMAGES_DIR = "data/nsd/images"

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"Already downloaded: {local_path}")
        return
        
    print(f"Downloading {url} to {local_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(local_path)) as t:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                t.update(len(chunk))
                f.write(chunk)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(COCO_IMAGES_DIR, exist_ok=True)
    
    # 1. Download Experimental Design (contains trial mappings)
    expdesign_url = f"{NSD_S3_BASE}/nsddata/experiments/nsd/nsd_expdesign.mat"
    expdesign_path = os.path.join(DATA_DIR, "nsd_expdesign.mat")
    download_file(expdesign_url, expdesign_path)
    
    # 2. Download Stimulus Info (CSV) for mapping NSD ID -> COCO URL
    stim_csv_url = f"{NSD_S3_BASE}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    stim_csv_path = os.path.join(DATA_DIR, "nsd_stim_info_merged.csv")
    download_file(stim_csv_url, stim_csv_path)
    
    # 3. Download Subject 01 NSDGeneral ROI Mask
    roi_url = f"{NSD_S3_BASE}/nsddata/ppdata/subj01/func1mm/roi/nsdgeneral.nii.gz"
    roi_path = os.path.join(DATA_DIR, "nsdgeneral.nii.gz")
    download_file(roi_url, roi_path)
    
    # 4. Download Subject 01, Session 1 fMRI Betas (2.8 GB)
    betas_url = f"{NSD_S3_BASE}/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz"
    betas_path = os.path.join(DATA_DIR, "betas_session01.nii.gz")
    download_file(betas_url, betas_path)
    
    print("\nExtracting Session 1 trial IDs...")
    # Load expdesign.mat (contains 1-indexed trial orders for all sessions)
    mat = sio.loadmat(expdesign_path)
    
    # Subject 1 is index 0. `subjectim` gives the NSD image IDs shown to this subject.
    # `masterordering` gives the 1-indexed trial order (but it is flattened for all sessions/subjects)
    # The first 750 trials correspond to Session 1.
    
    # It is easier to get the exact 750 NSD IDs for Subject 01, Session 01.
    # subjectim (73000 NSD IDs for all subjects). Subj01 uses the first row.
    subj01_images = mat['subjectim'][0] # length 10000 unique images
    
    # The actual order shown in the scanner is in masterordering (30000 trials for each subject)
    # The trial indices are 1-based, pointing to the 10000 subjectim array.
    session_1_trials = mat['masterordering'][0][:750] # first 750 trials for subj01
    
    # Get the global NSD IDs (1-indexed in MATLAB, subtract 1 for python indexing)
    session_1_nsd_ids = []
    for trial_idx in session_1_trials:
        # trial_idx is 1-indexed index into subj01_images
        global_nsd_id = subj01_images[trial_idx - 1] 
        session_1_nsd_ids.append(global_nsd_id)
        
    print(f"Extracted {len(session_1_nsd_ids)} NSD IDs for Session 1.")
    
    # Load the CSV to map NSD ID to COCO URL
    print("Loading stimulus info CSV...")
    df = pd.read_csv(stim_csv_path)
    
    print(f"Downloading {len(session_1_nsd_ids)} MS COCO images for Session 1...")
    for i, nsd_id in enumerate(tqdm(session_1_nsd_ids, desc="Images")):
        # NSD ID is 1-indexed in the matlab file and matches the NSD ID in the CSV (where nsdId is 0-indexed!)
        # Wait, let's check the CSV structure.
        row = df.iloc[nsd_id - 1] # nsdId in MATLAB is 1-73000. In python it's 0-72999
        coco_url = row['cocoUrl']
        
        # Save image locally as trial_0000.jpg, trial_0001.jpg etc to match fMRI sequence
        img_path = os.path.join(COCO_IMAGES_DIR, f"trial_{i:04d}.jpg")
        if not os.path.exists(img_path):
            try:
                urllib.request.urlretrieve(coco_url, img_path)
            except Exception as e:
                print(f"Failed to download {coco_url}: {e}")

    print("\n✅ Phase 4 Data Download Complete!")

if __name__ == "__main__":
    main()
