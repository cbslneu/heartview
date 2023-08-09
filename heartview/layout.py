from dash import html, dcc
from heartview import default
import dash_bootstrap_components as dbc
import dash_uploader as du
import uuid

layout = html.Div(id = 'main', children = [

    # OFFCANVAS
    dbc.Offcanvas(children = [
        dcc.Store(id = 'memory-load', storage_type = 'memory'),
        dcc.Store(id = 'memory-db', storage_type = 'memory'),
        dcc.Store(id = 'progress-clear', storage_type = 'memory'),
        html.Span(className = 'h5',
                  children = ['Welcome to HeartView']),
        html.P(children = [
            'Explore signal quality metrics for ambulatory cardiovascular '
            'data collected with devices such as the Empatica E4 or Actiwave '
            'Cardio.']),
        html.H4(children = [
            'Load Data',
            html.I(className = 'fa-regular fa-circle-question',
                   style = {'marginLeft': '5px'},
                   id = 'load-data-help')],
                style = {'display': 'flex', 'alignItems': 'center'}),
        dbc.Tooltip(
            'Valid file types: .edf (Actiwave); .zip (E4); .csv (other ECG '
            'devices).',
            target = 'load-data-help'),
        du.Upload(id = 'dash-uploader',
                  text = 'Select File...',
                  filetypes = ['EDF', 'edf', 'zip', 'csv'],
                  upload_id = uuid.uuid4(),
                  text_completed = '',
                  cancel_button = True,
                  default_style = {'lineHeight': '50px',
                                   'minHeight': '50px',
                                   'borderRadius': '5px'}),
        html.Div(id = 'file-check', children = []),
        html.H4(children = ['Load a Configuration File']),
        dcc.Dropdown(id = 'configs-dropdown',
                     options = default._get_configs()),
        html.Div(id = 'setup-data', hidden = True, children = [
            html.H4(children = [
                'Setup Data',
                html.I(className = 'fa-regular fa-circle-question',
                       style = {'marginLeft': '5px'},
                       id = 'data-var-help')],
                style = {'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip('Set your sampling rate and/or map the headers in '
                        'your file to the corresponding data variables.',
                        target = 'data-var-help'),
            html.Span('Sampling Rate: '),
            dcc.Input(id = 'sampling-rate', value = 1000, type = 'number',
                      style = {'display': 'inline',
                               'marginLeft': '3px'}),
            html.Div(id = 'data-variables', children = [
                html.P('Variables:', className = 'variables'),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-1',
                                     placeholder = 'Timestamp',
                                     options = ['<Var>', '<Var>']),
                        id = 'setup-timestamp'),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-2',
                                     placeholder = 'ECG',
                                     options = ['<Var>', '<Var>']),
                        id = 'setup-cardio'),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-3',
                                     placeholder = 'X',
                                     options = ['<Var>', '<Var>'])),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-4',
                                     placeholder = 'Y',
                                     options = ['<Var>', '<Var>'])),
                    dbc.Col(
                        dcc.Dropdown(id = 'data-type-dropdown-5',
                                     placeholder = 'Z',
                                     options = ['<Var>', '<Var>'])),
                ], className = 'setup-data-dropdowns')
            ]),
        ]),
        html.Div(id = 'segment-data', hidden = True, children = [
            html.H4(children = [
                'Segment Data',
                html.I(className = 'fa-regular fa-circle-question',
                       style = {'marginLeft': '5px'},
                       id = 'seg-data-help')],
                style = {'display': 'flex', 'alignItems': 'center'}),
            dbc.Tooltip(
                'This sets the size of the windows into which your data will '
                'be segmented; by default, 60 seconds.',
                target = 'seg-data-help'),
            html.Span('Window Size (sec): '),
            dcc.Input(id = 'seg-size', value = 60, type = 'number')
        ]),
        html.Div(id = 'ecg-filters', hidden = True, children = [
            html.H4('Apply ECG Filters', id = 'header-filter'),
            dcc.Checklist(id = 'filter-selector', options = [
                {'label': 'Baseline Wander & Muscle Noise',
                 'value': 'baseline-muscle'},
                {'label': 'Powerline Interference',
                 'value': 'powerline'}])
        ]),
        html.Div(children = [
            html.Button('Run', n_clicks = 0, id = 'run-data', style = {
                'width': '100px', 'marginTop': '20px'}, disabled = True),
            html.Button('Save', n_clicks = 0, id = 'configure', style = {
                'width': '100px', 'marginLeft': '10px', 'marginTop': '20px'},
                        disabled = False)
        ]),

        # Configuration File Exporter
        dbc.Modal(id = 'config-modal', is_open = False, children = [
            dbc.ModalHeader(dbc.ModalTitle('Export Configuration')),
            dbc.ModalBody(children = [
                html.Div(id = 'config-description', children = [
                    html.Div('Save your parameters in a JSON configuration '
                             'file.'),
                    html.Div(children = [
                        html.Span('Filename: '),
                        dcc.Input(id = 'config-filename', value = 'config1',
                                  type = 'text'),
                        html.Span(' .json')], style = {'marginTop': '20px'})
                ]),
                html.Div(id = 'config-check', children = [
                    html.I(className = 'fa-solid fa-circle-check',
                       style = {'color': '#63e6be', 'fontSize': '26pt'}),
                    html.P('Exported to: /configs', style = {'marginTop': '5px'})
                ], hidden = True)
            ]),
            dbc.ModalFooter(children = [
                html.Div(id = 'config-modal-btns', children = [
                    html.Button('Configure', n_clicks = 0, id = 'config-btn'),
                    dbc.Button('Cancel', n_clicks = 0, id = 'close-config1')],
                         hidden = False),
                html.Div(id = 'config-close-btn', children = [
                    dbc.Button('Done', n_clicks = 0, id = 'close-config2')],
                         hidden = True)
            ], style = {'display': 'inline'})
        ], backdrop = 'static', centered = True),

        # SQA Pipeline Progress Indicator
        dcc.Loading(children = [html.Div(id = 'progress-spin')],
                    type = 'default', color = '#ffa891'),
    ], id = 'offcanvas', style = {'width': 450}, is_open = True),

    # NAVIGATION BAR
    html.Div(className = 'banner', children = [
       dbc.Row(children = [
           dbc.Col(html.Img(src = 'assets/heartview-logo.png',
                            className = 'logo')),
           dbc.Col(html.Div(html.A('Documentation', href = '#')),
                   width = 3),
           dbc.Col(html.Div(html.A('Contribute', href = '#')),
                   width = 3)
       ], justify = 'between')
    ]),

    # MAIN DASHBOARD
    html.Div(className = 'app', children = [

        # Panel 1: Data Summary
        html.Div(className = 'data-summary', children = [
            html.Div(children = [
                html.H2('Data Summary'),
                html.I(className = 'fa-solid fa-pen-to-square',
                       id = 'reload-data'),
                dbc.Tooltip('Reload data file', target = 'reload-data')
            ], style = {'display': 'flex'}),
            html.Div(className = 'data-summary-divs', children = [
                html.Span('Device:', className = 'data-about'),
                html.Span('<device>', className = 'data-label', id = 'device')]),
            html.Div(className = 'data-summary-divs', children = [
                html.Span('Filename:', className = 'data-about'),
                html.Span('<file>', className = 'data-label', id = 'filename')]),
            html.Div(className = 'data-summary-divs', id = 'summary-table',
                     children = [default.blank_table()]),

            # Data Summary Exporter
            html.Div(children = [
                html.Button('Export Summary', id = 'export-summary', n_clicks = 0),
                dbc.Modal(id = 'export-modal', is_open = False, children = [
                    dbc.ModalHeader(dbc.ModalTitle('Export Summary')),
                    dbc.ModalBody(children = [
                        html.Div(id = 'export-description', children = [
                            html.Div('Download summary data as a Zip archive '
                                     'file (.zip) or Excel (.xlsx) file.'),
                            html.Div(children = [
                                html.I(
                                    className = 'fa-solid fa-download fa-bounce'),
                                dcc.RadioItems(['Zip', 'Excel'], inline = True,
                                               id = 'export-type')
                            ], style = {'textAlign': 'center'})
                        ]),
                        html.Div(id = 'export-confirm', children = [
                            html.I(className = 'fa-solid fa-circle-check',
                                   style = {'color': '#63e6be', 'fontSize': '26pt'}),
                            html.P('Exported to: /downloads', style = {'marginTop': '5px'})
                        ], hidden = True)
                    ]),
                    dbc.ModalFooter(children = [
                        html.Div(id = 'export-modal-btns', children = [
                            html.Button('OK', n_clicks = 0, id = 'ok-export'),
                            dcc.Download(id = 'download-summary'),
                            dbc.Button('Cancel', n_clicks = 0,
                                       id = 'close-export'),
                        ]),
                        html.Div(id = 'export-close-btn', children = [
                            dbc.Button('Done', n_clicks = 0,
                                       id = 'close-export2')],
                                 hidden = True)
                    ], style = {'display': 'inline'})
                ], backdrop = 'static', centered = True)
            ], style = {'paddingTop': '20px'})], style = {'padding': '15px'}),

        # Panel 2: Expected-to-Missing Beats
        html.Div(className = 'qa', children = [
            html.H2('Expected-to-Missing Beats'),
            html.Div(className = 'figs', children = [
                dcc.Graph(id = 'peaks',
                          figure = default.blank_fig('pending'),
                          style = {'height': '268px'})
            ]),
        ]),

        # Bottom Panel: Raw ECG/BVP, IBI, and ACC Plots
        html.Div(className = 'raw-plots', children = [
            html.Div(className = 'graph-settings', children = [
                html.Button(html.Span('ECG', id = 'data-type'),
                            id = 'load-raw', n_clicks = 0),
                html.Button('IBI', id = 'load-ibi', n_clicks = 0),
                html.Button('ACC', id = 'load-acc', n_clicks = 0),
                html.Span('Filter by segment:', className = 'slider-label'),
                html.Div(className = 'slider', children = [
                    dcc.RangeSlider(min = 1, max = 10, step = 1,
                                    value = [5, 7], marks = None,
                                    tooltip = {'placement': 'bottom',
                                               'always_visible': True},
                                    id = 'segment-range-slider')])
            ]),
            html.Div(className = 'figs', children = [
                dcc.Loading(type = 'circle', color = '#3a4952', children = [
                    dcc.Graph(id = 'raw-data',
                              figure = default.blank_fig('pending'),
                              style = {'height': '300px'})
                ])
            ])
        ]),
    ]),

    # FOOTER
    html.Div(className = 'bottom', children = [
        html.Span('Created by the '),
        html.A(href = 'https://github.com/cbslneu', children = [
            'Computational Behavioral Science Lab'], target = 'new'),
        html.Span(', Northeastern University')
    ])
])