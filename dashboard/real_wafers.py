"""Presentation of saved real-data artifacts; no extraction or model fitting."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

CATEGORY_NAMES = {
    'none': 'Normal', 'Center': 'Center', 'Donut': 'Donut',
    'Edge-Loc': 'Edge localized', 'Edge-Ring': 'Edge ring', 'Loc': 'Localized',
    'Near-full': 'Near full', 'Random': 'Random', 'Scratch': 'Scratch',
}
SPLIT_NAMES = {'grouped': 'Lot-held-out', 'metadata': 'Official split'}
COLOR = '#356a91'


def category_name(label):
    return CATEGORY_NAMES.get(label, label)


def feature_name(name):
    words = {'nw': 'NW', 'ne': 'NE', 'sw': 'SW', 'se': 'SE', 'std': 'SD', 'x': 'X', 'y': 'Y'}
    label = ' '.join(words.get(word, word) for word in name.split('_'))
    return label[0].upper() + label[1:]


def yield_chart(frame):
    counts, edges = np.histogram(frame.loc[frame.total_dies > 0, 'yield_rate'], bins=20, range=(0, 1))
    data = pd.DataFrame({'Yield': edges[:-1], 'Bin end': edges[1:], 'Wafers': counts})
    return alt.Chart(data).mark_bar(color=COLOR).encode(
        x=alt.X('Yield:Q', title='Passing dies / tested dies', bin='binned',
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%', values=[0, .2, .4, .6, .8, 1])),
        x2='Bin end:Q', y=alt.Y('Wafers:Q', axis=alt.Axis(format=',d', tickMinStep=1)),
        tooltip=[alt.Tooltip('Yield:Q', title='From', format='.0%'),
                 alt.Tooltip('Bin end:Q', title='To', format='.0%'), alt.Tooltip('Wafers:Q', format=',d')],
    ).properties(height=280)


def confusion_chart(report):
    records = [
        {'Actual': category_name(actual), 'Predicted': category_name(predicted), 'Wafers': int(count)}
        for actual, row in zip(report['labels'], report['confusion_matrix'])
        for predicted, count in zip(report['labels'], row)
    ]
    order = [category_name(label) for label in report['labels']]
    base = alt.Chart(pd.DataFrame(records)).encode(
        x=alt.X('Predicted:N', sort=order, axis=alt.Axis(labelAngle=-35, labelLimit=140, titlePadding=12)),
        y=alt.Y('Actual:N', sort=order, axis=alt.Axis(labelLimit=140)),
        tooltip=['Actual:N', 'Predicted:N', alt.Tooltip('Wafers:Q', format=',d')],
    )
    cells = base.mark_rect().encode(color=alt.Color('Wafers:Q', scale=alt.Scale(scheme='blues'), legend=None))
    threshold = max(max(row) for row in report['confusion_matrix']) * .55
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text('Wafers:Q', format=',d'),
        color=alt.condition(alt.datum.Wafers > threshold, alt.value('white'), alt.value('#22313f')),
    )
    return (cells + labels).properties(height=380)


def render_evaluation(report):
    st.subheader('Defect classification results')
    split_name = SPLIT_NAMES.get(report['split'], report['split'])
    st.caption(
        f"{split_name} · {report['train_rows']:,} training / {report['test_rows']:,} test wafers · "
        f"{report['train_lots']:,} training / {report['test_lots']:,} test lots · seed {report['seed']}."
    )
    if report['split'] == 'grouped':
        st.caption('Whole lots held out using a seeded split; this is separate from the official Training/Test assignment.')
    else:
        st.caption(f"Official Training/Test assignment; {report['purged_training_rows']:,} training rows removed to prevent shared-lot leakage.")
    st.caption(f"Lot-leakage audit: {report['lot_overlap']} shared lots between training and test.")
    comparison = pd.DataFrame([
        {'Model': model, 'Accuracy': scores['accuracy'], 'Macro F1': scores['macro_f1'], 'Weighted F1': scores['weighted_f1']}
        for model, scores in [('Random Forest', report['metrics']), ('Most frequent baseline', report['baseline'])]
    ])
    st.dataframe(comparison, hide_index=True, width='stretch', column_config={
        'Model': st.column_config.TextColumn(width='medium'),
        **{name: st.column_config.NumberColumn(format='%.3f') for name in ['Accuracy', 'Macro F1', 'Weighted F1']},
    })
    st.caption('Scores range from 0 to 1. Macro F1 gives each category equal weight; weighted F1 follows test-set prevalence.')
    test_counts = report['class_distribution']['test']
    majority = max(test_counts, key=test_counts.get)
    st.caption(f"Test-set imbalance: {category_name(majority)} accounts for {test_counts[majority] / report['test_rows']:.1%} of held-out wafers.")
    if report['baseline']['accuracy'] > report['metrics']['accuracy']:
        st.info('The majority baseline has higher accuracy on this split. Compare macro F1 and per-class recall before judging defect detection.')
    if 'Scratch' in report['per_class']:
        scratch = report['per_class']['Scratch']
        st.warning(f"Scratch recall: {scratch['recall']:.1%} on {int(scratch['support']):,} test wafers. This baseline misses many Scratch patterns.")
    if report['classes_missing_from_train'] or report['classes_missing_from_test']:
        st.warning('Some classes are missing from a partition; inspect split details before interpreting macro F1.')

    st.subheader('Confusion matrix')
    st.caption('Rows are actual labels; columns are predictions. Counts are annotated; hover for exact values. “Normal” is the source label `none`.')
    st.altair_chart(confusion_chart(report), width='stretch')
    with st.expander('Confusion matrix as a table'):
        names = [category_name(label) for label in report['labels']]
        matrix = pd.DataFrame(report['confusion_matrix'], index=names, columns=names)
        matrix.index.name = 'Actual / predicted'
        st.dataframe(matrix, width='stretch', height=390,
                     column_config={name: st.column_config.NumberColumn(width='small', format='%d') for name in names})

    st.subheader('Per-class precision and recall')
    rows = [{'Category': category_name(label), 'Precision': report['per_class'][label]['precision'],
             'Recall': report['per_class'][label]['recall'], 'F1': report['per_class'][label]['f1-score'],
             'Test wafers': int(report['per_class'][label]['support'])} for label in report['labels']]
    classes = pd.DataFrame(rows)
    bars = classes.melt(id_vars='Category', value_vars=['Precision', 'Recall'], var_name='Metric', value_name='Score')
    st.altair_chart(alt.Chart(bars).mark_bar().encode(
        y=alt.Y('Category:N', title=None, sort=classes.Category.tolist(), axis=alt.Axis(labelLimit=150)),
        yOffset='Metric:N', x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'), title='Score'),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['Precision', 'Recall'], range=[COLOR, '#a3bacb']), legend=alt.Legend(orient='top')),
        tooltip=['Category', 'Metric', alt.Tooltip('Score:Q', format='.1%')],
    ).properties(height=max(180, len(rows) * 43)), width='stretch')
    st.dataframe(classes, hide_index=True, width='stretch', column_config={
        **{name: st.column_config.ProgressColumn(format='%.3f', min_value=0, max_value=1) for name in ['Precision', 'Recall', 'F1']},
        'Category': st.column_config.TextColumn(width='medium'), 'Test wafers': st.column_config.NumberColumn(format='%d'),
    })
    with st.expander('Train/test class distribution and split details'):
        distribution = pd.DataFrame(report['class_distribution']).fillna(0).astype(int)
        distribution.index = distribution.index.map(category_name)
        st.dataframe(distribution, width='stretch')
        st.write(f"Classes absent from training: {', '.join(report['classes_missing_from_train']) or 'None'}. "
                 f"Classes absent from test: {', '.join(report['classes_missing_from_test']) or 'None'}.")


def render_real_wafers(root):
    st.header('Real WM-811K wafer-map analysis')
    st.caption('Public die pass/fail maps · Separate from the synthetic electrical-test workflow')
    with st.sidebar.expander('Real-data artifacts'):
        directory = Path(st.text_input('WM-811K artifact directory', str(root / 'data/processed/wm811k'), key='wafer_artifact_directory'))
    path = directory / 'features.parquet'
    if not path.is_file():
        st.info('Run python -m src.wafer_map_features.process_labeled_batch to prepare real-data artifacts.')
        return
    try:
        frame = pd.read_parquet(path)
        manifest_path = path.with_suffix('.manifest.json')
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        reports = {}
        for folder in ['model', 'model_official']:
            report_path = directory / folder / 'evaluation.json'
            if report_path.is_file():
                saved = json.loads(report_path.read_text())
                reports[f"{SPLIT_NAMES.get(saved['split'], saved['split'])} · {folder}"] = saved
        report = reports[st.selectbox('Evaluation split', list(reports))] if reports else None
        if report and report['lot_overlap'] != 0:
            st.error('Evaluation artifact reports shared training/test lots. Resolve the leakage audit before presenting these results.')
            return
        cards = st.columns(4)
        cards[0].metric('Labeled wafers in artifact', f'{len(frame):,}')
        for column, key, title in zip(cards[1:], ['accuracy', 'macro_f1', 'weighted_f1'], ['Accuracy', 'Macro F1', 'Weighted F1']):
            column.metric(title, (f"{report['metrics'][key]:.1%}" if key == 'accuracy' else f"{report['metrics'][key]:.3f}") if report else '—')
        if manifest.get('source_rows') is not None:
            st.caption(f"Source: {manifest['source_rows']:,} maps · {manifest['labeled_rows']:,} labeled · "
                       f"{manifest['source_rows'] - manifest['labeled_rows']:,} unlabeled (excluded from supervised learning).")
        if manifest.get('per_class_cap') is not None:
            st.warning('This artifact uses a capped per-class sample. Its prevalence and performance do not represent the full labeled population.')
        overview, evaluation, engineering = st.tabs(['Dataset & wafer maps', 'Model evaluation', 'Features & limitations'])
        with overview:
            left, right = st.columns(2)
            with left:
                st.subheader('Wafer yield distribution')
                st.altair_chart(yield_chart(frame), width='stretch')
                st.caption(f"Yield = passing / tested dies. {int(frame.total_dies.eq(0).sum()):,} zero-tested maps excluded.")
            with right:
                st.subheader('Defect-category distribution')
                counts = frame.failureType.value_counts()
                distribution = pd.DataFrame({'Category': counts.index.map(category_name), 'Wafers': counts.values})
                st.altair_chart(alt.Chart(distribution).mark_bar(color=COLOR).encode(
                    y=alt.Y('Category:N', sort='-x', title=None, axis=alt.Axis(labelLimit=150)),
                    x=alt.X('Wafers:Q', axis=alt.Axis(format=',d', tickCount=4)),
                    tooltip=['Category', alt.Tooltip('Wafers:Q', format=',d')],
                ).properties(height=280), width='stretch')
                top = counts.index[0]
                st.caption(f"Class imbalance: {category_name(top)} represents {counts.iloc[0] / len(frame):.1%} of this artifact. Normal is the labeled `none` class.")
            examples = path.with_suffix('.examples.npz')
            if examples.is_file():
                st.subheader('Example wafer maps')
                controls, preview = st.columns([1, 2])
                with np.load(examples, allow_pickle=False) as maps:
                    with controls:
                        category = st.selectbox('Defect category', maps.files, format_func=category_name)
                        st.caption('One saved example per category; examples illustrate patterns and are not held-out predictions.')
                        st.caption('White: empty · Green: passing die · Red: failing die')
                    with preview:
                        fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
                        ax.imshow(maps[category], cmap=ListedColormap(['white', '#6b9980', '#bc665e']), vmin=0, vmax=2, interpolation='nearest')
                        ax.set_title(category_name(category), fontsize=11)
                        ax.set_axis_off()
                        st.pyplot(fig, width='content')
                        plt.close(fig)
        with evaluation:
            if report:
                render_evaluation(report)
            else:
                st.info('No saved evaluation found. Prepare classifier artifacts to display measured results.')
        with engineering:
            if report:
                st.subheader('Feature importance')
                importance = pd.DataFrame({'Feature': [feature_name(name) for name in report['feature_importance']],
                                           'Importance': list(report['feature_importance'].values())})
                st.altair_chart(alt.Chart(importance).mark_bar(color=COLOR).encode(
                    y=alt.Y('Feature:N', sort='-x', title=None, axis=alt.Axis(labelLimit=280)),
                    x=alt.X('Importance:Q', title='Relative impurity importance', axis=alt.Axis(format='.0%')),
                    tooltip=['Feature', alt.Tooltip('Importance:Q', format='.2%')],
                ).properties(height=max(300, len(importance) * 23)), width='stretch')
                st.caption('Impurity importance is descriptive. Correlated features share credit; importance does not establish physical cause.')
            st.subheader('Engineering observations')
            st.write('Center/edge densities describe failure location; centroid, radial spread and adjacency describe geometry. These aggregate features can miss thin Scratch patterns. Class imbalance makes macro F1 and per-class recall essential alongside accuracy.')
            st.write('Lot-disjoint evaluation limits shared-lot leakage. Results depend on the selected split; a single holdout provides no confidence interval or external-fab validation. Visual defect labels do not establish physical root causes.')
            with st.expander('Artifact provenance'):
                st.caption('Source manifest for reproducibility; the dashboard reads saved artifacts and does not fit models.')
                if manifest:
                    st.json(manifest)
                st.code(str(directory), language=None)
    except (ValueError, KeyError, OSError) as exc:
        st.error(f'Unable to read real wafer artifacts: {exc}')
