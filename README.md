# HeartFrame

Interactive cardiac MRI exploration built from the ACDC notebooks in this repository. Explore a patient's original MRI, expert segmentation, physical 3D surfaces, and LV/RV measurements. Compare the notebook's saved Normal/DCM/HCM representatives at a common physical scale.

## Run locally

From this repository's root folder, using Python 3.11:

```bash
conda activate heartmotion
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Leave the terminal running. Streamlit opens the app in your browser.

Obtain ACDC separately from https://www.creatis.insa-lyon.fr/Challenge/acdc/ and place its training folder at `data/training`. Each `patientXXX` folder should contain `Info.cfg`, ED/ES MRI files, and their `_gt.nii.gz` reference masks. You can choose another training folder in the sidebar. No data is uploaded by this app.

## Behavior

- Uses ED/ES frame numbers from Info.cfg, never assumes frame01 is ED.
- Computes volumes from native reference-mask voxel counts and spacing, independently at each phase. EF is 100 Ã— (EDV âˆ’ ESV) / EDV.
- Uses the NIfTI affine for mesh coordinates; applies one shared translation across all structures and both phases. No anatomical registration across different patients is performed.
- Surface padding closes edges for rendering only; metrics use original masks. Displayed surfaces are visual approximations.
- Preserves the same 3D scale between phases and between comparison panels.
- Compare reads `outputs/representative_hearts.csv`, exported by notebook 03, and recomputes displayed measurements from local masks. Representatives are closest to their group's median LVEF.
- Missing/invalid geometry produces an error message. Missing or unusual chamber volumes produce review warnings. Use Reload data after replacing local files.

Original notebooks are unchanged. Raw cine playback, automated segmentation, indexed metrics, and trial review are future additions. The app does not estimate treatment effects: ED/ES are phases of one heartbeat, not treatment visits. ACDC groups are supplied dataset labels, partly defined using cardiac measurements, not independently predicted diagnoses.

## Dataset attribution

ACDC data is governed by its own CC BY-NC-SA 4.0 and accompanying noncommercial scientific research terms. Do not treat repository code licensing as relicensing the data or third-party materials. Download and review the original dataset license before use; raw images and masks are excluded from Git.

Required citation: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. â€œDeep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?â€ IEEE Transactions on Medical Imaging, 37(11), 2514â€“2525, 2018. https://doi.org/10.1109/TMI.2018.2837502

Research and education prototype, not a clinical diagnostic system.
