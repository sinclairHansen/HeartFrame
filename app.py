"""Heart in Motion: local ACDC viewer built from notebooks 0-3."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = '20260919_193223_583590'
GROUPS = {'NOR': 'Normal', 'DCM': 'Dilated Cardiomyopathy',
          'HCM': 'Hypertrophic Cardiomyopathy', 'MINF': 'Myocardial Infarction',
          'RV': 'Abnormal Right Ventricle'}
STRUCTURES = {'LV cavity': (3, '#ef657a'), 'RV cavity': (1, '#58b4e5'),
              'Myocardium': (2, '#eebd54')}


def read_patient_info(folder):
    return dict(line.strip().split(':', 1) for line in
                (Path(folder) / 'Info.cfg').read_text().splitlines() if ':' in line)


@st.cache_data(max_entries=24)
def load_patient(folder, source='Expert', predictions_root=''):
    folder = Path(folder)
    info = {k.strip(): v.strip() for k, v in read_patient_info(folder).items()}
    phases = {}
    for phase in ('ED', 'ES'):
        frame = int(info[phase])
        stem = folder / f'{folder.name}_frame{frame:02d}'
        image = nib.load(f'{stem}.nii.gz')
        if source == 'U-Net':
            mask_path = Path(predictions_root) / folder.name / f'{stem.name}_pred.nii.gz'
        else:
            mask_path = Path(f'{stem}_gt.nii.gz')
        if not mask_path.is_file():
            raise FileNotFoundError(f'Missing {source} mask: {mask_path}. No substitute mask was used.')
        mask_image = nib.load(mask_path)
        mask = mask_image.get_fdata()
        if image.shape != mask.shape or not np.allclose(image.affine, mask_image.affine):
            raise ValueError(f'{phase}: MRI and mask geometry do not match.')
        if not np.isin(mask, [0, 1, 2, 3]).all():
            raise ValueError(f'{phase}: unexpected segmentation labels.')
        if image.ndim != 3:
            raise ValueError('Expected a 3D ED/ES volume.')
        spacing = np.asarray(mask_image.header.get_zooms()[:3], dtype=float)
        if not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError('Invalid voxel spacing.')
        if not np.allclose(spacing, image.header.get_zooms()[:3], atol=1e-5):
            raise ValueError(f'{phase}: mask spacing does not match MRI spacing.')
        units = image.header.get_xyzt_units()[0]
        if units not in ('mm', 'unknown'):
            raise ValueError(f'Expected millimeter geometry, got {units}.')
        phases[phase] = dict(units=units, image=image.get_fdata(dtype=np.float32),
                             mask=mask.astype(np.uint8), spacing=spacing,
                             affine=mask_image.affine, frame=frame)
    return info, phases


def metrics(phases):
    rows = []
    for name, label in [('LV', 3), ('RV', 1)]:
        volumes = [float(np.count_nonzero(phases[p]['mask'] == label)
                         * np.prod(phases[p]['spacing']) / 1000) for p in ('ED', 'ES')]
        edv, esv = volumes
        rows.append({'Ventricle': name, 'EDV (mL)': edv, 'ESV (mL)': esv,
                     'SV (mL)': edv-esv, 'EF (%)': 100*(edv-esv)/edv if edv > 0 else np.nan})
    return pd.DataFrame(rows).set_index('Ventricle')


def prediction_files(folder, predictions_root):
    info = {k.strip(): v.strip() for k, v in read_patient_info(folder).items()}
    return [Path(predictions_root) / folder.name /
            f'{folder.name}_frame{int(info[phase]):02d}_pred.nii.gz' for phase in ('ED', 'ES')]


def build_meshes(phases):
    result = {}
    for phase, volume in phases.items():
        result[phase] = {}
        for name, (label, color) in STRUCTURES.items():
            mask = volume['mask'] == label
            if mask.any():
                vertices, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.uint8), 0.5)
                vertices = nib.affines.apply_affine(volume['affine'], vertices - 1)
                result[phase][name] = (vertices, faces, color)
    return result


@st.cache_data(max_entries=24)
def meshes(folder, source='Expert', predictions_root=''):
    _, phases = load_patient(folder, source, predictions_root)
    result = build_meshes(phases)
    points = [v for phase in result.values() for v, _, _ in phase.values()]
    # Frame both valid sources together, so toggling does not change scale or position.
    if predictions_root and all(p.is_file() for p in prediction_files(Path(folder), predictions_root)):
        other = 'Expert' if source == 'U-Net' else 'U-Net'
        try:
            _, other_phases = load_patient(folder, other, predictions_root)
            other_meshes = build_meshes(other_phases)
            points += [v for phase in other_meshes.values() for v, _, _ in phase.values()]
        except (OSError, ValueError):
            # An invalid optional source must not prevent viewing the selected source.
            pass
    if not points:
        return result, 50.0
    points = np.vstack(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    extent = max(1.0, float(np.max(np.abs(points-center)) * 1.1))
    for phase in result.values():
        for name, (v, f, c) in phase.items():
            phase[name] = (v-center, f, c)
    return result, extent


def comparison_table(reference, predicted):
    ref, pred = metrics(reference), metrics(predicted)
    return pd.DataFrame([
        {'Ventricle': chamber, 'Measurement': metric,
         'Expert': ref.loc[chamber, metric], 'U-Net': pred.loc[chamber, metric],
         'U-Net minus expert': pred.loc[chamber, metric]-ref.loc[chamber, metric]}
        for chamber in ref.index for metric in ref.columns
    ])


def dice_table(reference, predicted):
    records = []
    for phase in ('ED', 'ES'):
        for structure, (label, _) in STRUCTURES.items():
            ref, pred = reference[phase]['mask'] == label, predicted[phase]['mask'] == label
            denominator = int(ref.sum()) + int(pred.sum())
            records.append({'Phase': phase, 'Structure': structure,
                            'Dice': 2*int((ref & pred).sum())/denominator if denominator else np.nan})
    return pd.DataFrame(records)


def trace(mesh, name):
    v, f, color = mesh
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                     i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color,
                     name=name, opacity=0.45 if name == 'Myocardium' else 0.9,
                     hoverinfo='name', showlegend=True)


def scene(extent):
    return dict(**{f'{a}axis': dict(range=[-extent, extent], visible=False) for a in 'xyz'},
                aspectmode='cube', camera=dict(eye=dict(x=1.4, y=1.4, z=1.1)))


def draw_slice(image, mask, z, spacing, overlay):
    fig, ax = plt.subplots(figsize=(5, 5))
    bounds = (0, image.shape[0]*spacing[0], 0, image.shape[1]*spacing[1])
    low, high = np.percentile(image, [1, 99])
    ax.imshow(image[:, :, z].T, cmap='gray', origin='lower', extent=bounds, vmin=low, vmax=high)
    if overlay:
        ax.imshow(np.ma.masked_equal(mask[:, :, z].T, 0), origin='lower', extent=bounds,
                  cmap=ListedColormap(['black', '#58b4e5', '#eebd54', '#ef657a']),
                  vmin=0, vmax=3, interpolation='nearest', alpha=0.45)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)


def explore(folder, source='Expert', predictions_root='', split='training'):
    info, phases = load_patient(str(folder), source, predictions_root)
    st.subheader(f'{source} measurements and anatomy')
    if any(v['units'] == 'unknown' for v in phases.values()):
        st.caption('Spatial units are unspecified in the MRI header; millimeters are assumed for ACDC.')
    phase = st.radio('Cardiac phase', ['ED', 'ES'], horizontal=True,
                     format_func=lambda p: 'ED | Filled' if p == 'ED' else 'ES | Contracted')
    volume = phases[phase]
    table = metrics(phases)
    for name, row in table.iterrows():
        st.markdown(f'**{name} function**')
        for col, (label, value) in zip(st.columns(4), row.items()):
            col.metric(label, f'{value:.1f}' if np.isfinite(value) else 'Unavailable')
        if row['EDV (mL)'] <= 0 or row['ESV (mL)'] <= 0 or row['SV (mL)'] < 0:
            st.warning(f'{name}: missing or unusual chamber volumes. Review the selected masks.')
    st.caption('These measurements summarize ED and ES together; changing phase does not change EF.')
    left, right = st.columns([1, 1.5])
    with left:
        count = volume['image'].shape[2]
        z = st.slider('MRI slice', 1, count, (count+1)//2, key=f'slice_{folder.name}_{phase}')-1 if count > 1 else 0
        overlay = st.checkbox('Show segmentation overlay', value=True)
        draw_slice(volume['image'], volume['mask'], z, volume['spacing'], overlay)
        st.caption(f"Frame {volume['frame']} | Blue: RV | Gold: myocardium | Pink: LV")
    with right:
        selected = st.multiselect('3D structures', list(STRUCTURES), default=list(STRUCTURES))
        all_meshes, extent = meshes(str(folder), source, predictions_root)
        missing = [name for name in selected if name not in all_meshes[phase]]
        if missing:
            st.warning('Missing in this mask: ' + ', '.join(missing))
        fig = go.Figure([trace(all_meshes[phase][name], name) for name in selected if name in all_meshes[phase]])
        fig.update_layout(scene=scene(extent), height=520, margin=dict(l=0, r=0, t=0, b=0),
                          uirevision=folder.name, legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'{source} surfaces in physical coordinates. ED/ES share the same scale. Valid paired sources also share framing.')
    export = table.reset_index()
    export.insert(0, 'Patient', folder.name)
    export['Source'] = source
    export['Dataset'] = split
    export['Prediction folder'] = predictions_root if source == 'U-Net' else ''
    st.download_button('Download patient measurements', export.to_csv(index=False),
                       file_name=f'{folder.name}_{source}_metrics.csv', mime='text/csv')
    if source == 'U-Net':
        try:
            _, reference = load_patient(str(folder), 'Expert')
            differences = comparison_table(reference, phases)
            st.subheader('U-Net compared with expert masks')
            st.caption('Signed differences are U-Net minus expert. EF differences are percentage points, not relative percent errors.')
            st.dataframe(differences.round(3), hide_index=True)
            st.dataframe(dice_table(reference, phases).round(4), hide_index=True)
            st.download_button('Download expert/model comparison', differences.to_csv(index=False),
                               file_name=f'{folder.name}_expert_vs_unet.csv', mime='text/csv')
        except (OSError, ValueError) as exc:
            st.warning(f'Expert comparison unavailable: {exc}')


def compare(data_dir):
    csv_path = ROOT / 'outputs' / 'representative_hearts.csv'
    if not csv_path.exists():
        st.info('Run notebook 03 to save outputs/representative_hearts.csv.')
        return
    representatives = pd.read_csv(csv_path)
    records, prepared = [], []
    for _, row in representatives.iterrows():
        folder = data_dir / row['Patient']
        _, phases = load_patient(str(folder))
        mesh, extent = meshes(str(folder))
        prepared.append((row, mesh, extent))
        records.append({'Group': row['Group'], 'Patient': row['Patient'], **metrics(phases).loc['LV'].to_dict()})
    if not prepared:
        st.info('No representative patients in the CSV.')
        return
    n = len(prepared)
    titles = [f"{row['Group']} | {phase}<br>{row['Patient']}" for phase in ('ED', 'ES') for row, _, _ in prepared]
    fig = make_subplots(rows=2, cols=n, specs=[[{'type':'scene'} for _ in range(n)] for _ in range(2)], subplot_titles=titles)
    for col, (_, mesh, _) in enumerate(prepared, 1):
        for r, phase in enumerate(('ED', 'ES'), 1):
            if 'LV cavity' in mesh[phase]:
                fig.add_trace(trace(mesh[phase]['LV cavity'], 'LV cavity'), row=r, col=col)
    fig.update_scenes(**scene(max(item[2] for item in prepared)))
    fig.update_layout(height=760, showlegend=False, margin=dict(l=0, r=0, t=65, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(records).round(2), hide_index=True)
    st.caption('Saved notebook-03 representatives: closest to group median LVEF. All views share a physical scale. Dataset groups are supplied labels, not model predictions.')


def main():
    st.set_page_config(page_title='Heart in Motion', page_icon=':heart:', layout='wide')
    st.title('Heart in Motion')
    st.write('Explore cardiac MRI, ventricular anatomy, and function.')
    data_root = Path(st.sidebar.text_input('ACDC data folder', str(ROOT / 'data'))).expanduser()
    view = st.sidebar.radio('View', ['Explore', 'Compare'])
    if st.sidebar.button('Reload data'):
        st.cache_data.clear()
    try:
        if view == 'Compare':
            st.caption('Notebook-03 phenotype comparison | Training patients | Expert masks')
            if not (data_root / 'training').is_dir():
                st.info('Place training patients under the ACDC data folder to use Compare.')
            else:
                compare(data_root / 'training')
        else:
            split = st.sidebar.selectbox('Dataset', ['testing', 'training'],
                format_func=lambda x: 'Testing (final holdout)' if x == 'testing' else 'Training (includes validation)')
            data_dir = data_root / split
            run_dir = Path(st.sidebar.text_input('U-Net run folder',
                str(ROOT / 'outputs' / 'unet' / DEFAULT_RUN))).expanduser()
            output_split = 'test' if split == 'testing' else 'validation'
            predictions_root = str(run_dir / output_split / 'predictions')
            st.sidebar.caption(f'Prediction folder: {predictions_root}')
            source = st.sidebar.radio('Segmentation', ['Expert', 'U-Net'])
            only_predicted = st.sidebar.checkbox('Only patients with both predictions', value=(source == 'U-Net'),
                                                 key=f'filter_{source}_{split}')
            folders = sorted(p for p in data_dir.glob('patient*') if (p / 'Info.cfg').is_file())
            if not folders:
                st.info(f'No patients found in {data_dir}. Choose another dataset or check the ACDC data folder.')
                return
            available = [p for p in folders if all(f.is_file() for f in prediction_files(p, predictions_root))]
            st.sidebar.caption(f'{len(available)} / {len(folders)} patients have ED and ES predictions.')
            if only_predicted:
                folders = available
            if not folders:
                st.info('No paired predictions found. Check the run folder and finish NB4 evaluation, or uncheck the prediction filter to view expert masks.')
                return
            folder = st.sidebar.selectbox('Patient', folders, format_func=lambda p: p.name,
                                          key=f'patient_{split}_{source}_{only_predicted}')
            info = read_patient_info(folder)
            code = info.get('Group', '').strip()
            st.caption(f'{folder.name} | {GROUPS.get(code, code)} | {source}')
            if source == 'U-Net':
                st.info(f'Precomputed U-Net prediction | Run {run_dir.name} | {output_split} outputs. No training or live inference runs in this app.')
                missing = [f.name for f in prediction_files(folder, predictions_root) if not f.is_file()]
                if missing:
                    st.warning('Prediction unavailable: ' + ', '.join(missing) + '. Select Expert to view reference masks.')
                    return
            explore(folder, source, predictions_root, split)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        st.error(f'Could not display this case: {exc}')
    st.caption('Research and education prototype. Expert masks or precomputed U-Net segmentations; no diagnostic predictions. ACDC: Bernard et al., IEEE TMI (2018), doi:10.1109/TMI.2018.2837502.')


if __name__ == '__main__':
    main()
