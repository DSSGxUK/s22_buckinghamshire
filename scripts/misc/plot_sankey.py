import os
from codetiming import Timer

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go

from src import constants as c
from src import data_utils as d

ID = 'debug'
EVENTS_DF_PATH = os.path.join(c.RESULTS_DIR, 'events.csv')
TRANSITION_DF_PATH =  os.path.join(c.RESULTS_DIR, f'transitions_{ID}.csv')
LABELS_DF_PATH = os.path.join(c.RESULTS_DIR, f'labels_{ID}.csv')
CREATE_TRANSITION_DFS = True
SANKEY_PATH = os.path.join(c.PLOTS_DIR, f'sankey_{ID}.png')


def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    return result

def mask_last(x):
    result = np.ones_like(x)
    result[-1] = 0
    return result

def build_transition_df2(
    filtered_events_df
):
    filtered_events_df = filtered_events_df[filtered_events_df['transition'] == 'start']
    print(f'We have {len(filtered_events_df)} events.')
    
    with Timer(text="created groups in {:.8f}s"):
        upn_event_groups = filtered_events_df.groupby([c.UPN])[c.UPN]
    with Timer(text="filtered out sources in {:.8f}s"):
        source_df = filtered_events_df.loc[upn_event_groups.transform(mask_last).astype(bool)]
    with Timer(text="filtered out targets in {:.8f}s"):
        target_df = filtered_events_df.loc[upn_event_groups.transform(mask_first).astype(bool)]
    
    
    transition_df = pd.concat([
        source_df[['name']].reset_index(drop=True),
        target_df[['name']].reset_index(drop=True),
    ], axis=1)
    transition_df.columns = ['source', 'target']
    transition_df['weight'] = 1
    
    with Timer(text="filtered out targets in {:.8f}s"):
        transition_df = transition_df \
            .groupby(by=['source', 'target']) \
            .sum() \
            .reset_index()
    
    labels = pd.concat([
            transition_df['source'], 
            transition_df['target']
        ]).unique()
    
    transition_df = transition_df.replace(
        {label: i for i, label in enumerate(labels)}
    )
    
    return transition_df, labels

def hex_to_rgb(hex_color, alpha=1.):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)

def with_alpha(rgb, alpha):
    return rgb[:3] + (alpha,)

def get_node_colors(num_labels, link_srcs, node_alpha=0.8, link_alpha=0.4, color_palette=px.colors.qualitative.Dark24):
    colors = [hex_to_rgb(col) for col in color_palette]
    num_colors = len(colors)
    repd_colors = (colors * (num_labels // num_colors + 1))[:num_labels]
    node_colors = [with_alpha(c, alpha=node_alpha) for c in repd_colors]
    link_colors = [with_alpha(node_colors[src], alpha=link_alpha) for src in link_srcs]
    
    return node_colors, link_colors

def rgba_colors_to_str(colors):
    return [f'rgba{c}' for c in colors]

def build_sankey(transition_df, labels, min_weight=10):
    transition_df = transition_df[transition_df['weight'] >= min_weight]
    
    node_colors, link_colors = get_node_colors(
        num_labels=len(labels),
        link_srcs = transition_df['source'],
    )
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 50,
            thickness = 15,
            label = labels,
            color = rgba_colors_to_str(node_colors)

        ),
        link = dict(
            source = transition_df['source'], # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = transition_df['target'],
            value = transition_df['weight'],
            color = rgba_colors_to_str(link_colors)
        ))])
    fig.update_layout(title_text="Trajectories of Students in Buckinghamshire", font_size=10)
    
    return fig

if __name__ == '__main__':
    single_upns_df = pd.read_csv(c.SINGLE_UPNS_DATA_FP)
    single_upns_df.info()
    events_df = pd.read_csv(EVENTS_DF_PATH)
    events_df.info()
    
    print(f'Computing filter on events')
    type_filter = [
        'post_16_activity',
        'compulsory_schooling'
    ]
    events_df_mask = None
    if type_filter is not None:
        events_df_mask = events_df['type'].isin(type_filter)
    filtered_events_df = events_df[events_df_mask]
    print(f'Filtered out {len(events_df) - len(filtered_events_df)} events.')
    
    print(f'Creating joined events')
    with Timer(text="created joined events in {:.8f}s"):
        filtered_joined_events_df = pd.merge(filtered_events_df, single_upns_df, on=c.UPN, how='left')
    
    
    
    if CREATE_TRANSITION_DFS:
        print(f'Creating transition df')
        transition_df, labels = build_transition_df2(filtered_joined_events_df)
        transition_df.to_csv(TRANSITION_DF_PATH, index=False)
        pd.Series(labels).to_csv(LABELS_DF_PATH, index=False)
    transition_df = pd.read_csv(TRANSITION_DF_PATH)
    labels = pd.read_csv(LABELS_DF_PATH)
    
    print(f'Building Sankey Diagram')
    fig = build_sankey(transition_df, labels)
    fig.write_image(SANKEY_PATH)
    fig.show()