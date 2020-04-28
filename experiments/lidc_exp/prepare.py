from multiprocessing import Pool
from shutil import copyfile
import SimpleITK as sitk
import pylidc as pl
import numpy as np
import os, glob
import pynrrd
import time

# Import config
PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
import utils.exp_utils as utils
cf_file = utils.import_module("cf", "configs.py")
cf = cf_file.configs()


def prepare(pid):
    lidc_path = os.path.join(os.getenv("MDT_DATASETS_DIR", "LIDC-IDRI")) # LIDC normalized public dataset, see: wiki.cancerimagingarchive.net
    # Get scan from pylidc
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    print("processing:", scan.patient_id)
    vol_shape = scan.to_volume().shape
    # Cluster the annotations for the scan, and grab one.
    nodules = scan.cluster_annotations()
    nodule_ix = 0
    for nodule_anns in nodules:
        # Build 50% consensus mask
        cmask, cbbox, _ = consensus(nodule_anns, clevel=0.5)
        cmask_full = np.zeros(vol_shape)
        cmask_full[cbbox] = cmask
        # Load header from NRRD
        scan_path = glob.glob(os.path.join(lidc_path, pid, "*", "*", f"{pid}_CT.nrrd"))[0]
        header = nrrd.read_header(scan_path)
        # Write consensus to nrrd
        cmask_full = np.swapaxes(cmask_full, 0, 1)
        nodule_id = f"{pid}_{nodule_ix}"
        nrrd.write(os.path.join(cf.raw_data_dir, f"{nodule_id}.nrrd"), cmask_full, header=header)
        nodule_ix = nodule_ix + 1
        # Write scan nrrd
        copyfile(scan_path, os.path.join(cf.raw_data_dir, f"{pid}_CT.nrrd"))


def prepare_nodules():
    # TODO
    f = open(os.path.join(cf.root_dir, "characteristics.csv"), "w+")
    f.close()


if __name__ == "__main__":
    
    start_time = time.time()
    
    # Prepare LIDC dataset using multiprocessing
    pool = Pool(processes=os.cpu_count())
    p1 = pool.map(pp_patient, enumerate(paths))
    pool.close()
    pool.join()
    
    # Write nodules characteristics to csv file
    prepare_nodules()
    
    print(f"------ Ellapsed time : {time.time() - start_time} (s) ------")
