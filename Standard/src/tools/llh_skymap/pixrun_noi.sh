#!/bin/bash

#SBATCH --mail-type=end
#SBATCH --mail-user=caowy@mail.ustc.edu.cn

source activate 3ML
# python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/20210305-20230731_trans_fromhsc.root -r /data/home/cwy/Science/3MLWCDA/data/WCDA_DR_psf.root #WCDA

# python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/20210305-20230731_trans_fromhsc.root -r /data/home/cwy/Science/3MLWCDA/data/DR_ihep_MK2_newoldDRpsf.root #WCDA

# python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/WCDA_20240131_out.root -r /data/home/cwy/Science/3MLWCDA/data/DR_ihep_20240131_Crabpsf_mc.root #WCDA

# python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky_WCDA.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/20240731_hsc_out.root -r /data/home/cwy/Science/3MLWCDA/data/DR_ihep_20240131_hscpsf_mc.root #WCDA

python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky_KM2A.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/KM2A_20240731_xsq_out.root -r /data/home/cwy/Science/3MLWCDA/data/KM2A_DR_xsq.root #WCDA

# python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky_KM2A.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/KM2A_20240131_xsq1389.root -r /data/home/cwy/Science/3MLWCDA/data/KM2A_DR_20240131.root #KM2A

# python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/KM2A_all_final.root -r /data/home/cwy/Science/3MLWCDA/data/KM2A_DR_all.root #KM2A

# python3.9 /data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky.py --area ${no} -m /data/home/cwy/Science/3MLWCDA/data/KM2A1234full_skymap_rcy.root -r /data/home/cwy/Science/3MLWCDA/data/KM2A1234full_mcpsf_DRfinal.root #KM2A_new