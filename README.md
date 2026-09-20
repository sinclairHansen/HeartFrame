# HeartFrame

Explore cardiac MRI slices, expert and U-Net segmentations, interactive 3D ventricular anatomy, and measurements of cardiac function in a local browser app.

**MRI | segmentation | 3D reconstruction | ventricular measurements | expert/model comparison**

HeartFrame is a research and education prototype, not a clinical diagnostic system. It reconstructs the labeled ventricular cavities and myocardium, not the entire heart.

## Choose your setup

| Goal | What you need |
| --- | --- |
| Explore MRI and expert segmentations | App dependencies and downloaded ACDC images/reference masks |
| Compare the saved Normal/DCM/HCM examples | Training data and the included `outputs/representative_hearts.csv` |
| View U-Net results in the app | Matching local MRI data and precomputed `_pred.nii.gz` masks |
| Train a model and generate predictions | The above, PyTorch, a notebook environment, and NB4 |

**No PyTorch installation or model training is required to run the app.** The app reads saved masks; it does not run live inference. Raw MRI data, model weights, and prediction files are not bundled in this repository. A fresh clone cannot show patient images until you download ACDC, and cannot show U-Net results until you generate or obtain compatible predictions.

## 1. Get the code

Install [Git](https://git-scm.com/downloads) and either [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or Python 3.11. Then open a terminal:

```bash
git clone https://github.com/sinclairHansen/HeartFrame.git
cd HeartFrame
```

Alternatively, choose **Code Download ZIP** on GitHub, extract it, and open a terminal in the extracted folder containing `app.py` and `requirements.txt`.

The project was developed on macOS. Instructions below also cover Windows and Linux, but those platforms have not been verified end to end. App viewing needs no GPU. Model training benefits from a supported GPU and can be slow on CPU; no fixed training time is guaranteed.

## 2. Create an environment and install app dependencies

Choose **one** environment method.

### Option A: Conda (same workflow as development)

```bash
conda create -n heartframe python=3.11 pip -y
conda activate heartframe
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `heartmotion` already exists, skip the create command. Use `python -m pip`, not bare `pip`, to install into the interpreter you are actually running.

### Option B: Python venv

On macOS/Linux, with Python 3.11 installed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows Command Prompt, with Python 3.11 installed:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. If local policy blocks activation, use Command Prompt or invoke `.\.venv\Scripts\python.exe` directly instead of changing system policy.

Check the installation:

```bash
python -c "import sys, streamlit, nibabel, plotly, skimage; print(sys.executable); print('App imports OK')"
```

Dependencies are version ranges, not a locked environment. Exact versions can differ between installs.

## 3. Download and place ACDC data

1. Open the official [ACDC dataset page](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
2. Follow its download/access instructions and review the dataset terms. Download the **training images and reference masks**. Testing images and reference masks are also needed for final test evaluation and the Testing expert view.
3. Extract the archives. Keep the individual `.nii.gz` files compressed; NiBabel reads them directly.
4. Arrange the patient folders as shown below. If extraction creates an extra wrapper directory, move the `training` and `testing` folders into `data`.

| Path relative to the repository | Contents |
| --- | --- |
| `data/training/patient001/Info.cfg` | ED/ES frame numbers and patient metadata |
| `data/training/patient001/patient001_frame01.nii.gz` | Example phase MRI filename |
| `data/training/patient001/patient001_frame01_gt.nii.gz` | Corresponding expert mask |
| `data/training/patient001/...` | Other phase files, using their actual frame numbers |
| `data/testing/patient101/Info.cfg` | Example testing patient metadata |
| `data/testing/patient101/...` | Its MRI and reference masks |

The filenames above illustrate the layout: **do not rename phases to frame01**. The app reads ED/ES frame numbers from `Info.cfg` and requires both phase images and the masks for the selected source. The `_4d.nii.gz` cine file is not needed by the current app.

Data can live outside the repository. In the app, set **ACDC data folder** to the parent containing `training` and/or `testing`, not to an individual patient or to `training` itself. NB4 expects `data/training` beneath the project root unless you edit its configuration.

## 4. Launch the app

From the repository root with your environment active:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally [http://localhost:8501](http://localhost:8501). Leave the terminal running. Stop the server with **Ctrl+C**. For subsequent sessions, activate the same environment, return to the project root, and run the command again.

For your first run without predictions:

1. Select **View Explore**.
2. Select **Dataset’ Training (includes validation)**. The app initially defaults to Testing, so change this if you downloaded only training data.
3. Select **Segmentation Expert**.
4. Leave **Only patients with both predictions** unchecked.
5. Select a patient, move through slices, rotate the 3D view, and switch ED/ES.

Metric cards summarize both phases together; EF should not change when you change the displayed phase.

**Compare** always uses training patients and expert masks. The included representative CSV selects `patient068`, `patient008`, and `patient023`. Those patient folders must be available. You do not need to rerun notebooks to use the app when the data and included CSV are present.

## 5. Optional: generate U-Net predictions with NB4

### Install training and notebook tools

With the same environment active:

```bash
python -m pip install "torch>=2.2" ipykernel jupyterlab
python -m ipykernel install --user --name heartmotion --display-name "Python (heartframe)"
```

For NVIDIA/CUDA acceleration or platform-specific installation issues, use the matching command from the [official PyTorch installer](https://pytorch.org/get-started/locally/) in this environment. Supported Apple Silicon systems can use MPS. NB4 selects CUDA, then MPS, then CPU based on availability.

Open `notebooks/04_train_segmentation.ipynb` in VS Code with its Python and Jupyter extensions, and choose **Python (heartmotion)** as the notebook kernel. Alternatively:

```bash
python -m jupyterlab
```

Check the kernel before training:

```python
import sys
import torch
print(sys.executable)
print(torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('MPS:', torch.backends.mps.is_available())
```

### Run the notebook

1. Run sections 1â€“6 to configure data, split patients, prepare slices, and define the network.
2. Run section 7's overfit sanity check and inspect the alignment/predictions.
3. Run section 8 to train. Defaults are 192 Ã— 192 inputs, batch size 8, and up to 30 epochs with early stopping. Reduce batch size if memory runs out. Prepared slices are cached in RAM.
4. Run sections 9â€“12 to reload the best checkpoint, save validation predictions, compute errors, and inspect overlays/3D comparisons.
5. Keep `RUN_FINAL_TEST = False` while developing the model. Once choices are frozen, set it to `True` in a new cell immediately before section 13, then run section 13 with testing images and masks available. **Do not rerun the entire notebook just to change this flag**, as that creates another run and may retrain the model.

The full training dataset is split by patient and group into 80 training and 20 validation patients. Checkpoint selection uses validation loss. Final testing is separate; do not use it to tune the model and still describe it as untouched.

Every configuration execution creates a new timestamped run under `outputs/unet/`. Use the run printed by your notebook, not necessarily the author's original timestamp.

| Run output | Purpose |
| --- | --- |
| `best_unet.pt` | Selected model weights and configuration |
| `config.json`, `patient_split.csv` | Settings and patient assignments |
| `training_history.csv` | Epoch losses, monitoring Dice, and times |
| `validation/predictions/<patient>/..._pred.nii.gz` | Native-grid validation masks |
| `test/predictions/<patient>/..._pred.nii.gz` | Native-grid test masks, after section 13 |
| `validation/` or `test/` CSV reports | Per-patient Dice, clinical errors, and review flags |

A training interruption preserves the best checkpoint from completed epochs. Section 9 can load it by setting `CHECKPOINT_TO_LOAD`. Rerunning section 8 starts training from scratch; it is not optimizer-state resume. Exact reproduction of the author's model requires the same checkpoint; retraining can produce different results.

## 6. View predictions in the app

In **Explore**, set **U-Net run folder** to the timestamped run directory itself, for example:

```text
/your/project/HeartFrame/outputs/unet/20260919_193223_583590
```

**Do not enter the `test/predictions` subfolder in this field.** The app adds the appropriate subpath:

| Dataset selection | MRI folder | Prediction subfolder under the selected run |
| --- | --- | --- |
| Testing | `data/testing` | `test/predictions` |
| Training | `data/training` | `validation/predictions` |

Choose **Segmentation U-Net**. The patient filter can limit the list to cases with both ED and ES predictions. A prediction filename must match the MRI frame, replacing `.nii.gz` with `_pred.nii.gz`, and live in a matching patient folder.

The app updates the MRI overlay, meshes, and metrics together and displays expert/model differences and Dice. EF differences are **percentage points**. Missing predictions are not replaced with expert masks. If files were regenerated while the app was open, click **Reload data**.

The viewer needs only saved predictions, not `best_unet.pt` or PyTorch. There is currently no bundled pretrained checkpoint, one-click prediction download, or live-inference endpoint. Compatible masks must be generated locally or supplied separately under the applicable data terms.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` after installing | Run `python -c "import sys; print(sys.executable)"` and `python -m pip --version`. In a notebook check `sys.executable`; select the right kernel. Install there with `%pip install torch`, then restart the kernel if necessary. |
| No patients found | Check the ACDC data parent path, selected dataset, and extra extraction directories. A training-only download requires choosing Training. |
| No paired predictions | Check the run folder, selected dataset, and both ED/ES prediction filenames. Disable the prediction filter and choose Expert to explore without a model. |
| NB4 cannot find the project | Open the notebook from the repository or its `notebooks` folder, with `data/training` present, or edit its path configuration. |
| Geometry mismatch | Use NB4's native-grid exported masks and their matching source MRI. Do not manually resize or rename unrelated predictions. |

## Acknowledgments and dataset terms

Inspired by [Brainchop](https://github.com/neuroneural/brainchop)'s interactive MRI segmentation and visualization experience.

ACDC data is supplied separately under its [dataset terms](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), including the accompanying CC BY-NC-SA 4.0 and noncommercial scientific research conditions. Consult the downloaded `LICENSE_TERMS.md` and `MANDATORY_CITATION.md`. Code licensing does not relicense the dataset or third-party materials.

Required ACDC citation:

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?" *IEEE Transactions on Medical Imaging*, 37(11), 2514â€“2525, 2018. [doi:10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502).
