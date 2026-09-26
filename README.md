# HeartFrame

I built HeartFrame at HackMIT 2026 to explore how different cardiac conditions affect the way the heart fills and contracts. I trained a 2D U-Net in PyTorch to segment cardiac MRI slices, then built an interactive Streamlit app to bring those results into 3D. Users can explore ventricular anatomy, compare the model’s predictions with expert annotations, and see how ventricular volumes change between filling and contraction.

The live app can be viewed here: [HeartFrame](https://heartframe.streamlit.app)

I hope to soon have a write-up on my website on everything I learned for the project and how it works!


## Acknowledgments and dataset terms

Inspired by [Brainchop](https://github.com/neuroneural/brainchop)'s interactive MRI segmentation and visualization experience.

ACDC data is supplied separately under its [dataset terms](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), including the accompanying CC BY-NC-SA 4.0 and noncommercial scientific research conditions. Consult the downloaded `LICENSE_TERMS.md` and `MANDATORY_CITATION.md`. Code licensing does not relicense the dataset or third-party materials.

Required ACDC citation:

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?" *IEEE Transactions on Medical Imaging*, 37(11), 2514â€“2525, 2018. [doi:10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502).
