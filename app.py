"""HeartFrame: cardiac MRI segmentation, anatomy, and function."""
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
            raise FileNotFoundError(f'Missing {source} mask: {mask_path}.')
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
                # Match NB1: header spacing defines physical dimensions in MRI voxel axes.
                # Do not apply the affine afterward: that would introduce another scale.
                spacing = np.asarray(volume['spacing'], dtype=float)
                vertices, faces, _, _ = marching_cubes(
                    np.pad(mask, 1).astype(np.uint8), 0.5, spacing=tuple(spacing))
                vertices = vertices - spacing  # Undo the one-voxel padding in physical units.
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
                     name=name, opacity=0.30 if name == 'Myocardium' else 1.0,
                     flatshading=False, lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
                     hoverinfo='name', showlegend=True)


CAMERAS = {
    'Oblique': dict(x=1.35, y=1.35, z=0.95),
    'Along Y': dict(x=0, y=-2.0, z=0.1),
    'Along X': dict(x=2.0, y=0, z=0.1),
    'Along Z': dict(x=0.01, y=0.01, z=2.0),
}


def scene(extent, angle='Oblique', zoom=1.0):
    eye = {k: v / zoom for k, v in CAMERAS[angle].items()}
    return dict(**{f'{a}axis': dict(range=[-extent, extent], visible=False) for a in 'xyz'},
                aspectmode='cube', bgcolor='#0e1827',
                camera=dict(eye=eye, projection=dict(type='orthographic')))


def focus_meshes(meshes_by_phase, selected, framing=None):
    """One translation for both phases/all selected structures; never scale vertices."""
    framing = [meshes_by_phase] if framing is None else framing
    points = [mesh[0] for group in framing for phase in group.values()
              for name, mesh in phase.items() if name in selected]
    if not points:
        return {phase: {} for phase in meshes_by_phase}, 50.0
    points = np.vstack(points)
    center = (points.min(0) + points.max(0)) / 2
    radius = max(1.0, float(np.max(np.abs(points-center))) * 1.08)
    result = {phase: {name: (mesh[0]-center, mesh[1], mesh[2])
                      for name, mesh in items.items() if name in selected}
              for phase, items in meshes_by_phase.items()}
    return result, radius


def view_controls(prefix):
    with st.expander('View controls'):
        a, b, c = st.columns(3)
        angle = a.selectbox('View angle', list(CAMERAS), key=f'{prefix}_angle')
        zoom = b.slider('Zoom', 0.8, 1.8, 1.0, 0.1, key=f'{prefix}_zoom')
        height = c.slider('Viewer height', 480, 960, 680, 40, key=f'{prefix}_height')
        if st.button('Reset rotations', key=f'{prefix}_reset'):
            st.session_state[f'{prefix}_revision'] = st.session_state.get(f'{prefix}_revision', 0) + 1
    revision = f'{angle}_{zoom}_{st.session_state.get(f"{prefix}_revision", 0)}'
    return angle, zoom, height, revision


def style_figure(fig, extent, angle, zoom, height, revision):
    fig.update_scenes(**scene(extent, angle, zoom))
    fig.update_layout(height=height, autosize=True, paper_bgcolor='#0e1827',
                      font=dict(color='#e7edf5', size=14), showlegend=False,
                      margin=dict(l=8, r=8, t=60, b=8), uirevision=revision)
    fig.update_annotations(font=dict(color='#e7edf5', size=17))
    return fig


def phase_trace(mesh, name, phase, overlay=False):
    result = trace(mesh, name)
    result.update(name=f'{name} · {phase}')
    if overlay:
        result.update(color='#80c9ed' if phase == 'ED' else '#ff7890',
                      opacity=0.16 if phase == 'ED' else 1.0)
    return result


def contraction_figure(items, selected, mode, extent, angle, zoom, height, revision):
    """items = [(display title, shared-coordinate ED/ES meshes), ...]."""
    both = mode == 'Side by side'
    rows, cols = (2, len(items)) if both and len(items) > 1 else (1, 2 if both else len(items))
    if both and len(items) == 1:
        titles = ['End diastole · Filled', 'End systole · Contracted']
    elif both:
        titles = [f'{title} · {phase}' for phase in ('ED', 'ES') for title, _ in items]
    else:
        titles = [title for title, _ in items]
    fig = make_subplots(rows=rows, cols=cols,
                        specs=[[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)],
                        subplot_titles=titles, horizontal_spacing=0.015, vertical_spacing=0.06)
    for col, (_, data) in enumerate(items, 1):
        phases = ('ED', 'ES') if mode in ('Side by side', 'Contraction overlay') else (('ED',) if mode == 'Filled (ED)' else ('ES',))
        for phase in phases:
            if both:
                r, c = ((1, 1 if phase == 'ED' else 2) if len(items) == 1
                        else (1 if phase == 'ED' else 2, col))
            else:
                r, c = 1, col
            for name in selected:
                if name in data[phase]:
                    fig.add_trace(phase_trace(data[phase][name], name, phase,
                                             overlay=(mode == 'Contraction overlay')), row=r, col=c)
    return style_figure(fig, extent, angle, zoom, height*(1.7 if rows == 2 else 1), revision)


def show_figure(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={'displaylogo': False, 'scrollZoom': True,
                            'toImageButtonOptions': {'format': 'png', 'scale': 2}})


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
    _, phases = load_patient(str(folder), source, predictions_root)
    table = metrics(phases)
    st.subheader('See the contraction')
    selected = st.multiselect('3D structures', list(STRUCTURES), default=['LV cavity'],
                              help='Start with the LV cavity to compare filling and contraction.')
    mode = st.radio('3D display', ['Side by side', 'Contraction overlay', 'Filled (ED)', 'Contracted (ES)'],
                    horizontal=True)
    angle, zoom, height, revision = view_controls('explore')
    all_meshes, _ = meshes(str(folder), source, predictions_root)
    framing = [all_meshes]
    if predictions_root and all(f.is_file() for f in prediction_files(folder, predictions_root)):
        try:
            other, _ = meshes(str(folder), 'Expert' if source == 'U-Net' else 'U-Net', predictions_root)
            framing.append(other)
        except (OSError, ValueError):
            pass
    focused, extent = focus_meshes(all_meshes, selected, framing)
    if not selected:
        st.info('Select a structure to view its 3D anatomy.')
    else:
        for phase in ('ED', 'ES'):
            missing = [name for name in selected if name not in focused[phase]]
            if missing:
                st.warning(f'{phase}: missing ' + ', '.join(missing))
        fig = contraction_figure([(folder.name, focused)], selected, mode, extent,
                                  angle, zoom, height, f'{folder.name}_{selected}_{revision}')
        show_figure(fig, 'explore_anatomy')
        if mode == 'Contraction overlay':
            st.caption('Blue translucent shell: filled (ED) · Pink: contracted (ES). Two measured phases, shown together.')
        else:
            st.caption('Drag to rotate · Scroll to zoom · Expand the chart for fullscreen. Both phases share one physical scale.')
    with st.expander('Geometry details'):
        st.caption('Models use MRI voxel axes scaled by header spacing, matching NB1. '
                   'They are not displayed in scanner-world orientation.')
        details = []
        for phase_name, data in phases.items():
            header_spacing = np.asarray(data['spacing'])
            affine_spacing = np.linalg.norm(data['affine'][:3, :3], axis=0)
            label_mask = data['mask'] == 3
            indices = np.argwhere(label_mask)
            dimensions = ((indices.max(0) - indices.min(0) + 1) * header_spacing
                          if len(indices) else np.zeros(3))
            details.append({'Phase': phase_name,
                            'Header spacing (mm)': ', '.join(f'{v:.3f}' for v in header_spacing),
                            'Affine axis lengths': ', '.join(f'{v:.3f}' for v in affine_spacing),
                            'LV bounding size (mm)': ' × '.join(f'{v:.1f}' for v in dimensions)})
            if not np.allclose(header_spacing, affine_spacing, rtol=1e-3, atol=1e-4):
                st.warning(f'{phase_name}: header spacing and affine axis lengths differ. '
                           'Display and volume calculations both use header spacing; verify the source geometry before interpreting absolute dimensions.')
        st.dataframe(pd.DataFrame(details), hide_index=True)
    st.subheader('Cardiac function')
    for name, row in table.iterrows():
        st.markdown(f'**{name}**')
        for col, (label, value) in zip(st.columns(4), row.items()):
            col.metric(label, f'{value:.1f}' if np.isfinite(value) else 'Unavailable')
        if row['EDV (mL)'] <= 0 or row['ESV (mL)'] <= 0 or row['SV (mL)'] < 0:
            st.warning(f'{name}: missing or unusual chamber volumes. Review the selected masks.')
    with st.expander('About these measurements'):
        st.write('EDV: end-diastolic volume · ESV: end-systolic volume · SV: stroke volume · EF: ejection fraction.')
        st.caption('Measurements summarize both phases. The display controls change the anatomy shown, not the measurements.')
        if any(v['units'] == 'unknown' for v in phases.values()):
            st.caption('ACDC spatial units are interpreted as millimeters where the image header does not specify units.')

    # MRI has its own full-width section, rather than competing with the 3D viewer.
    st.subheader('Inspect the MRI')
    phase = st.radio('MRI phase', ['ED', 'ES'], horizontal=True)
    volume = phases[phase]
    count = volume['image'].shape[2]
    controls, image_column = st.columns([1, 3])
    with controls:
        z = st.slider('MRI slice', 1, count, (count+1)//2,
                      key=f'slice_{folder.name}_{phase}')-1 if count > 1 else 0
        overlay = st.checkbox('Show segmentation overlay', value=True)
        st.caption(f"Frame {volume['frame']} · {source}")
        st.caption('Blue: RV · Gold: myocardium · Pink: LV')
    with image_column:
        draw_slice(volume['image'], volume['mask'], z, volume['spacing'], overlay)

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
            st.subheader('Model performance')
            st.caption('Differences: U-Net minus expert. EF differences are in percentage points.')
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
        raw_mesh, _ = meshes(str(folder))
        # Tight framing uses only the displayed LV, with one center for ED and ES.
        mesh, extent = focus_meshes(raw_mesh, ['LV cavity'])
        prepared.append((row, mesh, extent))
        records.append({'Group': row['Group'], 'Patient': row['Patient'], **metrics(phases).loc['LV'].to_dict()})
    if not prepared:
        st.info('No representative patients in the CSV.')
        return
    summary = pd.DataFrame(records)
    mode = st.radio('Comparison display',
                    ['Contraction overlay', 'Filled (ED)', 'Contracted (ES)', 'Side by side'], horizontal=True)
    focus = st.selectbox('Focus', ['All phenotypes'] + summary['Group'].tolist(),
                         help='Choose one phenotype for a larger two-phase inspection.')
    angle, zoom, height, revision = view_controls('compare')
    extent = max(item[2] for item in prepared)
    chosen = prepared if focus == 'All phenotypes' else [item for item in prepared if item[0]['Group'] == focus]
    for column, (row, _, _) in zip(st.columns(len(chosen)), chosen):
        values = summary[summary['Patient'] == row['Patient']].iloc[0]
        with column:
            st.markdown(f"**{row['Group']}**")
            st.metric('LV ejection fraction', f"{values['EF (%)']:.1f}%")
            st.caption(f"{values['EDV (mL)']:.1f} mL filled → {values['ESV (mL)']:.1f} mL contracted")
    fig = contraction_figure([(row['Group'], mesh) for row, mesh, _ in chosen], ['LV cavity'],
                              mode, extent, angle, zoom, height, f'{focus}_{mode}_{revision}')
    show_figure(fig, 'phenotype_anatomy')
    if mode == 'Contraction overlay':
        st.caption('Blue translucent shell: ED · Pink: ES. All phenotypes share the same physical scale.')
    else:
        st.caption('Shared physical scale and starting view across all phenotypes and phases.')
    st.caption('Use View angle or Reset rotations to restore matching views after rotating individual panels.')
    # Connect visual contraction with the actual measured volume change.
    chart = go.Figure()
    for phase, field, color in [('Filled (ED)', 'EDV (mL)', '#80c9ed'),
                                ('Contracted (ES)', 'ESV (mL)', '#ff7890')]:
        chart.add_bar(name=phase, x=summary['Group'], y=summary[field], marker_color=color,
                      hovertemplate='%{x}<br>%{y:.1f} mL<extra>%{fullData.name}</extra>')
    chart.update_layout(title='Ventricular volume: filling to contraction', barmode='group',
                         yaxis_title='LV volume (mL)', height=340, margin=dict(t=55, b=20),
                         legend=dict(orientation='h', y=1.13))
    st.plotly_chart(chart, use_container_width=True, config={'displaylogo': False})
    with st.expander('Patient measurements and selection'):
        st.dataframe(summary.round(2), hide_index=True)
        st.write('Each case is closest to its group median left ventricular ejection fraction. '
                 'A single translation is applied to both phases; no heart is independently resized.')
        st.caption('Groups are ACDC reference labels. These are representative examples, not universal disease appearances. '
                   'The cavity view does not show myocardial wall thickness.')


def main():
    st.set_page_config(page_title='HeartFrame', page_icon=':heart:', layout='wide')
    st.markdown('<style>.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}</style>',
                unsafe_allow_html=True)
    st.title('HeartFrame')
    st.write('From cardiac MRI to 3D anatomy and function.')
    st.sidebar.title('HeartFrame')
    view = st.sidebar.radio('View', ['Explore', 'Compare'])
    with st.sidebar.expander('Data settings'):
        data_root = Path(st.text_input('ACDC data folder', str(ROOT / 'data'))).expanduser()
        run_dir = Path(st.text_input('U-Net run folder',
            str(ROOT / 'outputs' / 'unet' / DEFAULT_RUN))).expanduser()
        if st.button('Reload data'):
            st.cache_data.clear()
    try:
        if view == 'Compare':
            st.subheader('Compare cardiac phenotypes')
            st.caption('Training cohort · Expert segmentation')
            if not (data_root / 'training').is_dir():
                st.info('Place training patients under the ACDC data folder to use Compare.')
            else:
                compare(data_root / 'training')
        else:
            split = st.sidebar.selectbox('Dataset', ['testing', 'training'],
                format_func=lambda x: 'Testing (final holdout)' if x == 'testing' else 'Training (includes validation)')
            data_dir = data_root / split
            output_split = 'test' if split == 'testing' else 'validation'
            predictions_root = str(run_dir / output_split / 'predictions')
            source = st.sidebar.radio('Segmentation', ['Expert', 'U-Net'])
            only_predicted = st.sidebar.checkbox('Only patients with both predictions', value=(source == 'U-Net'),
                                                 key=f'filter_{source}_{split}')
            folders = sorted(p for p in data_dir.glob('patient*') if (p / 'Info.cfg').is_file())
            if not folders:
                st.info(f'No patients found in {data_dir}. Choose another dataset or check the ACDC data folder.')
                return
            available = [p for p in folders if all(f.is_file() for f in prediction_files(p, predictions_root))]
            st.sidebar.caption(f'Model results available for {len(available)} of {len(folders)} patients.')
            # Keep the patient independently of source/filter widget identities.
            selection_key = f'selected_patient::{data_dir.resolve()}'
            widget_key = f'patient_picker::{data_dir.resolve()}'
            all_folders = {p.name: p for p in folders}
            previous = st.session_state.get(selection_key)
            choices = [p.name for p in (available if only_predicted else folders)]
            pinned = previous in all_folders and previous not in choices
            if pinned:
                # Preserve the current patient even when a source-specific filter hides it.
                choices = sorted([*choices, previous])
            if not choices:
                st.info('No model results found. Check Data settings or turn off the prediction filter to explore expert segmentations.')
                return
            if previous not in choices:
                previous = choices[0]
            st.session_state[selection_key] = previous
            st.session_state[widget_key] = previous

            def remember_patient():
                st.session_state[selection_key] = st.session_state[widget_key]

            patient_id = st.sidebar.selectbox('Patient', choices, key=widget_key,
                                               on_change=remember_patient)
            folder = all_folders[patient_id]
            if pinned:
                st.sidebar.caption('Keeping your selected patient; model results are unavailable for this case.')
            info = read_patient_info(folder)
            code = info.get('Group', '').strip()
            st.caption(f'{folder.name} | {GROUPS.get(code, code)} | {source}')
            if source == 'U-Net':
                st.caption('Precomputed U-Net segmentation')
                with st.expander('Model details'):
                    st.caption(f'Run: {run_dir.name} · Evaluation split: {output_split}')
                missing = [f.name for f in prediction_files(folder, predictions_root) if not f.is_file()]
                if missing:
                    st.warning('Prediction unavailable: ' + ', '.join(missing) + '. Select Expert to view reference masks.')
                    return
            explore(folder, source, predictions_root, split)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        st.error(f'Could not display this case: {exc}')
    st.divider()
    st.caption('HeartFrame · Cardiac MRI exploration')
    with st.expander('About HeartFrame'):
        st.write('Explore expert and U-Net segmentations, ventricular anatomy, and cardiac function.')
        st.caption('Research and education prototype.')
        st.markdown('ACDC dataset: Bernard et al., IEEE TMI (2018). '
                    '[Dataset publication](https://doi.org/10.1109/TMI.2018.2837502)')


if __name__ == '__main__':
    main()
