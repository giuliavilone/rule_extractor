import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from common_functions import dataset_uploader
from c45_test import create_tree
from keras.models import load_model
import dash_cytoscape as cyto


def create_edges(tree_node_list, touched_nodes, color_name):
    ret = []
    for i in range(len(tree_node_list)):
        if tree_node_list[i] != -1:
            if i in touched_nodes and tree_node_list[i] in touched_nodes:
                ret += [{'data': {'source': str(i), 'target': tree_node_list[i]}, 'classes': 'orange'}]
            else:
                if len(touched_nodes) > 0:
                    ret += [{'data': {'source': str(i), 'target': tree_node_list[i]}, 'classes': 'greyed_out'}]
                else:
                    ret += [{'data': {'source': str(i), 'target': tree_node_list[i]}, 'classes': color_name}]
    return ret


def retrieve_fired_nodes(decision_tree, sample):
    node_indicator = decision_tree.decision_path(sample.reshape(1, -1))
    return list(node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]])


def create_node_labels(col_names, class_labels, decision_tree, separator=' <= '):
    values = [class_labels[np.where(arr[0] == np.amax(arr[0]))[0][0]] for arr in decision_tree.tree_.value]
    names = [str(col_names[decision_tree.tree_.feature[i]]) + separator + str(round(decision_tree.tree_.threshold[i], 2))
             if decision_tree.tree_.feature[i] != -2 else values[i] for i in range(len(values))]
    return names


def create_network(class_labels, col_names, decision_tree, sample=[], **kwargs):
    if len(sample) > 0:
        touched_nodes = retrieve_fired_nodes(decision_tree, sample)
    else:
        touched_nodes = []
    color_list = ['#fcc603', '#03c2fc', '#fc03db']
    ret = []
    names = create_node_labels(col_names, class_labels, decision_tree)
    for i in range(len(names)):
        if i in touched_nodes:
            ret += [{'data': {'id': str(i), 'label': names[i]}, 'classes': 'orange'}]
        else:
            if len(touched_nodes) > 0:
                ret += [{'data': {'id': str(i), 'label': names[i]}, 'classes': 'greyed_out'}]
            else:
                if names[i] in class_labels:
                    color_number = class_labels.index(names[i])
                    ret += [{'data': {'id': str(i), 'label': names[i]}, 'classes': 'color' + str(color_number)}]
                else:
                    ret += [{'data': {'id': str(i), 'label': names[i]}, 'classes': 'color_base'}]
    edges_left = create_edges(decision_tree.tree_.children_left, touched_nodes, 'green')
    edges_right = create_edges(decision_tree.tree_.children_right, touched_nodes, 'red')
    return ret + edges_left + edges_right


def node_style():
    color_list = ['#F923C3', '#FD611D', '#841DFD']
    node_styles = [{'selector': '.color' + str(i), 'style': {'background-color': color_list[i]}}
                   for i in range(len(color_list))]
    node_styles +=[{'selector': '.color_base', 'style': {'background-color': '#76C5C8'}}]
    return node_styles


# Interface
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
parameters = pd.read_csv('datasets-UCI/UCI_csv/summary.csv')
label_col = 'class'
data_path = 'datasets-UCI/UCI_csv/'
dataset_par = parameters.iloc[0]
X_train, X_test, _, _, labels, _, _ = dataset_uploader(dataset_par, data_path, cross_split=3, apply_smothe=False)
X_train, X_test = X_train[0], X_test[0]
column_height = '800px'
columns = X_train.columns.tolist()
model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                   + str(dataset_par['best_model']) + '.h5')
x_tot, y_tot, clf = create_tree(X_train, model)

graph_elements = create_network(labels, columns, clf)

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(children=[
    dbc.Row([
        dbc.Col(html.Div([
            cyto.Cytoscape(
                id='cytoscape-graph',
                layout={'name': 'cose'},
                style={'height': column_height},
                elements=graph_elements,
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {'content': 'data(label)'}
                    },
                    {
                        'selector': 'edge',
                        'style': {'curve-style': 'bezier'}
                    },
                    {
                        'selector': '.red',
                        'style': {'line-color': 'red',
                                  'target-arrow-color': 'red',
                                  'target-arrow-shape': 'triangle'}
                    },
                    {
                        'selector': '.green',
                        'style': {'line-color': 'green',
                                  'target-arrow-color': 'green',
                                  'target-arrow-shape': 'triangle'}
                    },
                    {
                        'selector': '.orange',
                        'style': {'background-color': 'orange',
                                  'shape': 'rectangle',
                                  'line-color': 'orange',
                                  'target-arrow-color': 'orange',
                                  'target-arrow-shape': 'orange'
                                  }
                    },
                    {
                        'selector': '.greyed_out',
                        'style': {'background-color': '#C6C6C6',
                                  'line-color': '#C6C6C6',
                                  'target-arrow-color': '#C6C6C6',
                                  'target-arrow-shape': '#C6C6C6'
                                  }
                    }] + node_style()
            )
            ]), width=7
        ),
        dbc.Col(html.Div([
            html.H4("Dataset: " + dataset_par['dataset'], style={'color': '#21618C'}),
            dcc.Tab(label='Fixed costs', children=[
                dash_table.DataTable(
                    id='datatable-interactivity',
                    columns=[{"name": i, "id": i} for i in X_test.columns],
                    data=X_test.to_dict('records'),
                    filter_action='native',
                    page_size=30,
                    style_table={'height': column_height, 'overflowY': 'auto'},
                    fixed_rows={'headers': True},
                    editable=False,
                    row_selectable='single'
                )
            ])
        ]), width=5
        )
    ], align="centre", justify="centre", no_gutters=False)
])


@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_rows')
)
def update_styles(selected_rows=[]):
    if selected_rows != None:
        return [{
            'if': {'row_index': i},
            'background_color': '#D2F3FF'
        } for i in selected_rows]
    else:
        return None


@app.callback(
    Output('cytoscape-graph', "elements"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"),
    State('cytoscape-graph', "elements")
)
def update_graph(rows, derived_virtual_selected_rows, elements):
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
    dff = None if rows is None else pd.DataFrame(rows)
    if len(derived_virtual_selected_rows) > 0:
        select_data = dff.iloc[derived_virtual_selected_rows].to_numpy()
        elements = create_network(labels, columns, clf, sample=select_data)
    return elements


if __name__ == '__main__':
    app.run_server(debug=True)
