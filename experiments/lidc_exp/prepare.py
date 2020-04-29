from multiprocessing import Pool
from shutil import copyfile
from pathlib import Path
import SimpleITK as sitk
import os, sys, glob
import pylidc as pl
from pylidc.utils import consensus
import numpy as np
import nrrd
import time
import csv

# Import config
PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
import utils.exp_utils as utils
cf_file = utils.import_module("cf", "configs.py")
cf = cf_file.configs()
lidc_path = os.path.join(os.getenv("MDT_DATASETS_DIR"), "LIDC-IDRI") # LIDC normalized public dataset, see: wiki.cancerimagingarchive.net


def prepare(pid):
    os.makedirs(os.path.join(cf.raw_data_dir, pid), exist_ok=True)
    # Get scan from pylidc
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    print("processing:", scan.patient_id)
    vol_shape = scan.to_volume().shape
    # Write scan nrrd
    scan_path = glob.glob(os.path.join(lidc_path, pid, "*", "*", f"{pid}_CT.nrrd"))[0]
    copyfile(scan_path, os.path.join(cf.raw_data_dir, pid, f"{pid}_CT.nrrd"))
    # Cluster the annotations for the scan, and grab one.
    nodules = scan.cluster_annotations()
    nodule_ix = 0
    for nodule_anns in nodules:
        # Build 50% consensus mask
        cmask, cbbox, _ = consensus(nodule_anns, clevel=0.5)
        cmask_full = np.zeros(vol_shape)
        cmask_full[cbbox] = cmask
        # Load header from NRRD
        header = nrrd.read_header(scan_path)
        # Write consensus to nrrd
        cmask_full = np.swapaxes(cmask_full, 0, 1)
        nodule_id = f"{pid}_nod_{nodule_ix}"
        nrrd.write(os.path.join(cf.raw_data_dir, pid, f"{nodule_id}.nrrd"), cmask_full, header=header)
        nodule_ix = nodule_ix + 1


def prepare_nodules():
    print("Writing to csv file")
    f = open(os.path.join(cf.root_dir, "characteristics.csv"), "w+")
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["patient_id", "nodule_id", "malignancy"])
    for pid in os.listdir(lidc_path):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        nodules = scan.cluster_annotations()
        for nodule_ix, nodule_anns in enumerate(nodules):
            nodule_id = f"{pid}_nod_{nodule_ix}"
            malignancy = nodule_anns[0].malignancy
            print(f"nodule: {nodule_id}, malignancy: {malignancy}")
            writer.writerow([pid, nodule_id, malignancy])
    f.close()


if __name__ == "__main__":
    start_time = time.time()
    # Prepare LIDC dataset using multiprocessing
    os.makedirs(cf.raw_data_dir, exist_ok=True)
    pids = os.listdir(lidc_path)
    pool = Pool(processes=os.cpu_count())
    p1 = pool.map(prepare, pids)
    pool.close()
    pool.join()
    # Write nodules characteristics to csv file
    prepare_nodules()
    print(f"------ Ellapsed time : {time.time() - start_time} (s) ------")
