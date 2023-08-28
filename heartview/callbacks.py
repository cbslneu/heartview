from dash import html, Input, Output, State, ctx, callback
from dash.exceptions import PreventUpdate
from heartview.pipeline import ECG, PPG, SQA, ACC
from heartview import default
from os import listdir, stat, path
from math import ceil
from time import sleep
import dash_uploader as du
import zipfile
import pandas as pd

def get_callbacks(app):
    """Attach callback functions to the dashboard app."""

    # ============================= DATA UPLOAD ===============================
    du.configure_upload(app, './temp', use_upload_id = True)
    @du.callback(
        output = [
            Output('file-check', 'children'),
            Output('run-data', 'disabled'),
            Output('configure', 'disabled'),
            Output('memory-load', 'data'),
        ],
        id = 'dash-uploader'
    )
    def db_get_file_types(filenames):
        """Save the data type to the local memory depending on the file
        type."""
        temp = './temp'
        session = [s for s in listdir(temp) if
                   path.isdir(f'{temp}/{s}') and s != 'cfg'][0]
        file = sorted(
            listdir(f'{temp}/{session}'),
            key = lambda t: -stat(f'{temp}/{session}/{t}').st_mtime)[0]
        filename = f'{temp}/{session}/{file}'

        if filenames[0].endswith(('edf', 'EDF')):
            if default._check_edf(filenames[0]) == 'ECG':
                file_check = [
                    html.I(className = 'fa-solid fa-circle-check',
                           style = {'color': '#63e6be', 'marginRight': '5px'}),
                    html.Span('Data loaded.')
                ]
                data = {'type': 'Actiwave',
                        'filename': filenames[0]}
                disable_run = False
                disable_configure = False
            else:
                file_check = [
                    html.I(className = 'fa-solid fa-circle-xmark',
                           style = {'color': '#cf0e22', 'marginRight': '5px'}),
                    html.Span('Invalid data type!')]
                data = 'invalid'
                disable_run = True
                disable_configure = True
        else:
            if filenames[0].endswith('zip'):
                z = zipfile.ZipFile(filename)
                if 'BVP.csv' not in z.namelist():
                    data = 'invalid'
                    file_check = [
                        html.I(className = 'fa-solid fa-circle-xmark',
                               style = {'color': '#cf0e22',
                                        'marginRight': '5px'}),
                        html.Span('Invalid data type!')
                    ]
                    disable_run = True
                    disable_configure = True
                else:
                    file_check = [
                        html.I(className = 'fa-solid fa-circle-check',
                               style = {'color': '#63e6be',
                                        'marginRight': '5px'}),
                        html.Span('Data loaded.')
                    ]
                    data = {'type': 'E4',
                            'filename': filename}
                    disable_run = False
                    disable_configure = False
            if filenames[0].endswith('csv'):
                file_check = [
                    html.I(className = 'fa-solid fa-circle-check',
                           style = {'color': '#63e6be', 'marginRight': '5px'}),
                    html.Span('Data loaded.')
                ]
                data = {'type': 'csv',
                        'filename': filename}
                disable_run = False
                disable_configure = False

        return [file_check, disable_run, disable_configure, data]

    # ==================== ENABLE CONFIGURATION UPLOAD ========================
    # === Toggle configuration uploader =======================================
    @app.callback(
        Output('config-upload-div', 'hidden'),
        Input('toggle-config', 'on'),
        prevent_initial_call = True
    )
    def db_enable_config_upload(toggle_on):
        """Display configuration file upload."""
        if toggle_on is True:
            hidden = False
        else:
            hidden = True
        return hidden

    # === Read JSON configuration file ========================================
    @du.callback(
        output = Output('config-memory', 'data'),
        id = 'config-uploader'
    )
    def db_get_config_file(cfg_file):
        configs = default._load_config(cfg_file[0])
        return configs

    # =================== POPULATE DATA VARIABLE DROPDOWNS ====================
    @app.callback(
        [Output('setup-data', 'hidden'),
         Output('segment-data', 'hidden'),
         Output('ecg-filters', 'hidden'),
         Output('data-variables', 'hidden'),
         Output('sampling-rate', 'value'),
         Output('data-type-dropdown-1', 'options'),
         Output('data-type-dropdown-2', 'options'),
         Output('data-type-dropdown-3', 'options'),
         Output('data-type-dropdown-4', 'options'),
         Output('data-type-dropdown-5', 'options'),
         Output('filter-selector', 'value'),
         Output('seg-size', 'value')],
        Input('memory-load', 'data'),
        Input('config-memory', 'data'),
        State('toggle-config', 'on')
    )
    def db_handle_csv_params(data, configs, toggle_config_on):
        """Output dropdown values according to uploaded CSV data."""
        loaded = ctx.triggered_id
        if loaded is None:
            raise PreventUpdate
        else:
            hide_setup = False
            hide_segsize = False

            if loaded == 'memory-load':
                if data['type'] != 'csv':
                    if data['type'] == 'Actiwave':
                        hide_data_vars = True
                        hide_filters = False
                        drop1, drop2, drop3, drop4, drop5 = (
                            [] for _ in range(5))
                        if toggle_config_on:
                            filters = configs['filters']
                            seg_size = configs['segment size']
                            fs = configs['sampling rate']
                        else:
                            filters = []
                            seg_size = 60
                            fs = 1000
                    else:
                        hide_filters = True
                        hide_setup = True
                        hide_data_vars = True
                        filters = []
                        drop1, drop2, drop3, drop4, drop5 = (
                            [] for _ in range(5))
                        if toggle_config_on:
                            seg_size = configs['segment size']
                            fs = configs['sampling rate']
                        else:
                            seg_size = 60
                            fs = 64
                else:
                    if toggle_config_on:
                        pass
                    else:
                        drop1 = default._get_csv_headers(data['filename'])
                        drop2 = default._get_csv_headers(data['filename'])
                        drop3 = default._get_csv_headers(data['filename'])
                        drop4 = default._get_csv_headers(data['filename'])
                        drop5 = default._get_csv_headers(data['filename'])
                        filters = []
                        seg_size = 60
                        fs = 1000
                        hide_data_vars = False
                        hide_filters = False

            if loaded == 'config-memory':
                device = configs['device']
                if device == 'E4':
                    hide_filters = True
                    hide_setup = True
                    hide_data_vars = True
                if device == 'Actiwave':
                    hide_data_vars = True

                filters = configs['filters']
                fs = configs['sampling rate']
                hide_data_vars = False
                hide_filters = False
                seg_size = configs['segment size']
                options = list(configs['headers'].values())
                drop1, drop2, drop3, drop4, drop5 = (options for _ in range(5))

            return [hide_setup, hide_segsize, hide_filters, hide_data_vars, fs,
                    drop1, drop2, drop3, drop4, drop5, filters, seg_size]

    # =================== TOGGLE EXPORT CONFIGURATION MODAL ===================
    @app.callback(
        [Output('config-download-memory', 'clear_data'),
         Output('config-modal', 'is_open'),
         Output('config-description', 'hidden'),
         Output('config-check', 'hidden'),
         Output('config-modal-btns', 'hidden'),
         Output('config-close-btn', 'hidden')],
        [Input('configure', 'n_clicks'),
         Input('close-config1', 'n_clicks'),
         Input('close-config2', 'n_clicks'),
         Input('config-download-memory', 'data'),
         State('config-modal', 'is_open')],
        prevent_initial_call = True
    )
    def toggle_config_modal(n, n1, n2, config_data, is_open):
        """Open and close the Export Configuration modal."""

        hide_config_desc = False  # show export fields
        hide_config_check = True
        hide_config_btns = False  # show 'configure' and 'cancel'
        hide_config_close = True

        if is_open is True:
            # If 'Cancel' or 'Done' is clicked
            if n1 or n2:
                print("closed modal")
                # Reset the content and close the modal
                return [True, not is_open,
                        hide_config_desc, hide_config_check,
                        hide_config_btns, hide_config_close]

            # If a configuration file was created and exported
            hide_config_desc = True
            hide_config_check = False
            hide_config_btns = True
            hide_config_close = False
            if config_data is not None:
                print(f'There is data: {config_data}')
                # Keep the modal open and show export confirmation
                return [True, is_open,
                        hide_config_desc, hide_config_check,
                        hide_config_btns, hide_config_close]
            else:
                return [False, is_open,
                        hide_config_desc, hide_config_check,
                        hide_config_btns, hide_config_close]

        else:
            # If 'Save' is clicked
            if n:
                if config_data is not None:
                    return [True, not is_open,
                            hide_config_desc, hide_config_check,
                            hide_config_btns, hide_config_close]
                else:
                    return [False, not is_open,
                            hide_config_desc, hide_config_check,
                            hide_config_btns, hide_config_close]

    # ====================== CREATE AND SAVE CONFIG FILE ======================
    @app.callback(
        [Output('config-file-download', 'data'),
         Output('config-download-memory', 'data')],
        [Input('config-btn', 'n_clicks'),
         State('memory-load', 'data'),
         State('sampling-rate', 'value'),
         State('seg-size', 'value'),
         State('data-type-dropdown-1', 'value'),
         State('data-type-dropdown-2', 'value'),
         State('data-type-dropdown-3', 'value'),
         State('data-type-dropdown-4', 'value'),
         State('data-type-dropdown-5', 'value'),
         State('filter-selector', 'value'),
         State('config-filename', 'value')],
        prevent_initial_call = True
    )
    def write_confirm_config(n, data, fs, seg_size, d1, d2, d3, d4, d5,
                             filters, filename):
        """Export the configuration file."""

        if n:
            device = data['type'] if data['type'] != 'csv' else 'Other'
            headers = {'Timestamp': d1, 'ECG': d2, 'X': d3, 'Y': d4, 'Z': d5}
            json_object = default._create_configs(
                device, fs, seg_size, filters, headers)
            download = {'content': json_object, 'filename': f'{filename}.json'}
            return [download, 1]

    # ============================ RUN PIPELINE ==============================
    @callback(
        output = Output('memory-db', 'data'),
        inputs = [
            Input('run-data', 'n_clicks'),
            State('memory-load', 'data'),
            State('sampling-rate', 'value'),
            State('data-type-dropdown-1', 'value'),
            State('data-type-dropdown-2', 'value'),
            State('data-type-dropdown-3', 'value'),
            State('data-type-dropdown-4', 'value'),
            State('data-type-dropdown-5', 'value'),
            State('seg-size', 'value'),
            State('filter-selector', 'value')
        ],
        background = True,
        running = [
            (Output('progress-bar', 'style'),
             {'visibility': 'visible'}, {'visibility': 'hidden'}),
            (Output('stop-run', 'hidden'), False, True),
            (Output('run-data', 'disabled'), True, False),
            (Output('configure', 'disabled'), True, False)
        ],
        cancel = [Input('stop-run', 'n_clicks')],
        progress = [
            Output('progress-bar', 'value'),
            Output('progress-bar', 'label')
        ],
        prevent_initial_call = True
    )
    def run_pipeline(set_progress, n, load_data, fs, d1, d2, d3, d4, d5,
                     seg_size, filters):
        """Read Actiwave Cardio, Empatica E4, or CSV-formatted data, save
        the data to the local memory, and load the progress spinner."""
        if n == 0:
            raise PreventUpdate
        else:
            total_progress = 6

            data_type = load_data['type']
            filename = load_data['filename']
            file = filename.split("/")[-1].split(".")[0]
            data = {}

            # Read uploaded data file
            if data_type == 'Actiwave' or data_type == 'csv':
                sleep(1)
                perc = (1 / total_progress) * 100
                set_progress((perc, f'{perc:.0f}%'))
                if data_type == 'Actiwave':
                    ecg, acc = ECG.read_actiwave(filename)
                    fs = ECG.get_fs(filename)
                else:
                    if (d3 is not None) \
                            & (d4 is not None) \
                            & (d5 is not None):
                        raw = pd.read_csv(
                            filename,
                            usecols = [d1, d2, d3, d4, d5]).rename(
                            columns = {d1: 'Timestamp',
                                       d2: 'ECG',
                                       d3: 'X',
                                       d4: 'Y',
                                       d5: 'Z'})
                        ecg = raw[['Timestamp', 'ECG']]
                        acc = raw[['Timestamp', 'X', 'Y', 'Z']]
                    else:
                        ecg = pd.read_csv(
                            filename,
                            usecols = [d1, d2]).rename(
                            columns = {d1: 'Timestamp',
                                       d2: 'ECG'})

                # Filter ECG signal
                sleep(1)
                perc = (2 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                if len(filters) <= 1:
                    if 'baseline-muscle' in filters:
                        ecg['Filtered'] = ECG.baseline_muscle_filter(
                            ecg['ECG'], 0.5, 45, fs)
                    else:
                        ecg['Filtered'] = ECG.powerline_int_filter(
                            ecg['ECG'], fs, 20, 60)
                else:
                    ecg['BM'] = ECG.baseline_muscle_filter(
                        ecg['ECG'], 0.5, 45, fs)
                    ecg['Filtered'] = ECG.powerline_int_filter(
                        ecg['BM'], fs, 20, 60)

                # Detect R peaks
                perc = (3 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                peaks_ix = ECG.detect_rpeaks(ecg, 'Filtered', fs)
                ecg.loc[peaks_ix, 'Peak'] = 1
                ecg.to_csv(f'./temp/{file}_ECG.csv', index = False)

                # Compute IBIs from peaks
                perc = (4 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                ibi = ECG.compute_ibis(ecg, 'Timestamp', peaks_ix)
                ibi.to_csv(f'./temp/{file}_IBI.csv', index = False)

                # Get second-by-second HR and IBI values
                perc = (5 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                interval_data = ECG.get_seconds(ecg, 'Peak', fs, seg_size)

            else:
                perc = (2 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                e4_data = PPG.preprocess_e4(filename)
                ibi, acc, bvp = e4_data['ibi'], e4_data['acc'], e4_data['bvp']
                fs = e4_data['fs']
                start_time = e4_data['start time']

                # Extract PPG peaks from IBIs
                perc = (3 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                peaks = PPG.get_e4_peaks(ibi, fs, start_time)
                ibi = peaks[~peaks['IBI'].isna()].reset_index(drop = True)

                # Get second-by-second HR and IBI values
                perc = (4 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))
                interval_data = PPG.get_e4_interval_data(peaks, seg_size)

                bvp.to_csv(f'./temp/{file}_BVP.csv', index = False)
                ibi.to_csv(f'./temp/{file}_IBI.csv', index = False)

                perc = (5 / total_progress) * 100
                set_progress((perc * 100, f'{perc:.0f}%'))

            # Pre-process acceleration data
            try:
                acc
            except NameError:
                pass
            else:
                acc['Magnitude'] = ACC.compute_magnitude(
                    acc['X'], acc['Y'], acc['Z'])
                acc.to_csv(f'./temp/{file}_ACC.csv', index = False)

            # Assess signal quality by segment
            peaks_by_seg = SQA.evaluate_peaks(interval_data, seg_size)
            peaks_by_seg.to_csv('./temp/peaks_by_segment.csv', index = False)
            metrics = SQA.compute_metrics(peaks_by_seg)
            metrics.to_csv('./temp/sqa_metrics.csv', index = False)
            perc = (6 / total_progress) * 100
            set_progress((perc * 100, f'{perc:.0f}%'))

            # Store data variables in memory
            data['type'] = data_type
            data['fs'] = fs
            data['filename'] = file
            data['interval data'] = interval_data.to_dict()

            return data

    # ===================== CONTROL DASHBOARD ELEMENTS ========================
    # === Toggle offcanvas ====================================================
    @app.callback(
        Output('offcanvas', 'is_open'),
        Input('reload-data', 'n_clicks')
    )
    def reload_data(n):
        """Open and close the offcanvas."""
        if n == 0:
            raise PreventUpdate
        else:
            return True
        
    # === Update segment range slider =========================================
    @app.callback(
        [Output('segment-range-slider', 'max'),
         Output('segment-range-slider', 'value')],
        Input('memory-db', 'data'),
    )
    def db_set_slider_range(data):
        """Set the range of the slider."""
        if data is None:
            raise PreventUpdate
        else:
            interval_data = pd.DataFrame.from_dict(data['interval data'])
            n_seg = int(interval_data['Segment'].iloc[-1])

            return [n_seg, [round(n_seg * 0.5), n_seg - 1]]

    # === Update plots and table ==============================================
    @app.callback(
        [Output('load-raw', 'style'),
         Output('load-ibi', 'style'),
         Output('load-acc', 'style'),
         Output('peaks', 'figure'),
         Output('device', 'children'),
         Output('filename', 'children'),
         Output('summary-table', 'children'),
         Output('raw-data', 'figure')],
        Input('memory-db', 'data'),
        Input('load-raw', 'n_clicks'),
        Input('load-ibi', 'n_clicks'),
        Input('load-acc', 'n_clicks'),
        Input('segment-range-slider', 'value'),
        State('seg-size', 'value'),
        State('segment-range-slider', 'value'),
        State('segment-range-slider', 'max')
    )
    def db_render_summary(data, raw_btn, ibi_btn, acc_btn,
                          slider, seg_size, selected_seg, slider_max):
        """Display the dashboard's summary visualizations and table."""
        if data is None:
            raise PreventUpdate
        else:
            file = data['filename']
            fs = int(data['fs'])
            seg_size = int(seg_size)
            sqa = pd.read_csv('./temp/sqa_metrics.csv')

            # Set the device name and get the total number of segments
            if data['type'] == 'E4':
                device = 'Empatica E4'
                bvp = pd.read_csv(f'./temp/{file}_BVP.csv')
                n_seg = ceil((len(bvp) / fs) / seg_size)
            else:
                if data['type'] == 'Actiwave':
                    device = 'Actiwave Cardio'
                else:
                    device = 'Other'
                ecg = pd.read_csv(f'./temp/{file}_ECG.csv')
                n_seg = ceil((len(ecg) / fs) / seg_size)

            # Create SQA plot and table
            exp2missing = SQA.plot_expected2missing(sqa, file)
            table = SQA.display_summary_table(sqa)

            # Style dashboard buttons for plots
            inactive = {'backgroundColor': '#47555e'}
            active = {'backgroundColor': '#313d44'}

            db_input = ctx.triggered_id if ctx.triggered_id else 'memory-db'

            # ACC Plots
            if db_input == 'load-acc':
                if f'{file}_ACC.csv' in listdir('./temp'):
                    acc = pd.read_csv(f'./temp/{file}_ACC.csv')
                    if slider is not [round(n_seg * 0.5), n_seg - 1] or \
                            selected_seg is not [round(n_seg * 0.5),
                                                 n_seg - 1]:
                        seg_num = slider[0]
                        seg_n = (slider[1] - slider[0]) * (seg_size / 60)
                        if seg_num == slider_max:
                            acc_plot = default.plot_signal(
                                acc, 'Timestamp', 'Magnitude', fs,
                                seg_size, seg_num, 1, 'acc')
                        else:
                            acc_plot = default.plot_signal(
                                acc, 'Timestamp', 'Magnitude', fs,
                                seg_size, seg_num, seg_n, 'acc')
                    else:
                        acc_plot = default.plot_signal(
                            acc, 'Timestamp', 'Magnitude', fs,
                            seg_size, round(n_seg * 0.5), 1, 'acc')
                    return [inactive, inactive, active,
                            None, device, file, table, acc_plot]
                else:
                    return [inactive, inactive, active, None, device, file,
                            table, default._blank_fig('none')]

            # IBI Plots
            elif db_input == 'load-ibi':
                ibi = pd.read_csv(f'./temp/{file}_IBI.csv')
                if slider is not [round(n_seg * 0.5), n_seg - 1] or \
                        selected_seg is not [round(n_seg * 0.5), n_seg - 1]:
                    seg_num = slider[0]
                    seg_n = (slider[1] - slider[0]) * (seg_size / 60)
                    if seg_num == slider_max:
                        ibi_plot = default.plot_signal(
                            ibi, 'Timestamp', 'IBI', 1,
                            seg_size, seg_num, 1, 'ibi')
                    else:
                        ibi_plot = default.plot_signal(
                            ibi, 'Timestamp', 'IBI', 1,
                            seg_size, seg_num, seg_n, 'ibi'
                    )
                else:
                    ibi_plot = default.plot_signal(
                        ibi, 'Timestamp', 'IBI', 1,
                        seg_size, round(n_seg * 0.5), 1, 'ibi')

                return [inactive, active, inactive,
                        None, device, file, table, ibi_plot]

            # ECG/BVP Plots
            else:
                if slider is not [round(n_seg * 0.5), n_seg - 1] or \
                        selected_seg is not [round(n_seg * 0.5), n_seg - 1]:
                    seg_num = slider[0]
                    seg_n = (slider[1] - slider[0]) * (seg_size / 60)
                    if data['type'] == 'E4':
                        bvp = pd.read_csv(f'./temp/{file}_BVP.csv')
                        if seg_num == slider_max:
                            raw_plot = default.plot_signal(
                                bvp, 'Timestamp', 'BVP', fs,
                                seg_size, seg_num, 1, 'bvp', 'Peak')
                        else:
                            raw_plot = default.plot_signal(
                                bvp, 'Timestamp', 'BVP', fs,
                                seg_size, seg_num, seg_n, 'bvp', 'Peak')
                    else:
                        ecg = pd.read_csv(f'./temp/{file}_ECG.csv')
                        if seg_num == slider_max:
                            raw_plot = default.plot_signal(
                                ecg, 'Timestamp', 'ECG', fs,
                                seg_size, seg_num, 1, 'ecg', 'Peak')
                        else:
                            raw_plot = default.plot_signal(
                                ecg, 'Timestamp', 'ECG', fs,
                                seg_size, seg_num, seg_n, 'ecg', 'Peak')
                else:
                    if data['type'] == 'E4':
                        bvp = pd.read_csv(f'./temp/{file}_BVP.csv')
                        raw_plot = default.plot_signal(
                            bvp, 'Timestamp', 'BVP', fs,
                            seg_size, round(n_seg * 0.5), 1, 'bvp', 'Peak')
                    else:
                        ecg = pd.read_csv(f'./temp/{file}_ECG.csv')
                        raw_plot = default.plot_signal(
                            ecg, 'Timestamp', 'ECG', fs,
                            seg_size, round(n_seg * 0.5), 1, 'ecg', 'Peak'
                        )

                return [active, inactive, inactive,
                        exp2missing, device, file, table, raw_plot]

    # === Open export summary modal ===========================================
    @app.callback(
        Output('export-modal', 'is_open'),
        [Input('export-summary', 'n_clicks'),
         Input('close-export', 'n_clicks'),
         Input('close-export2', 'n_clicks')],
        State('export-modal', 'is_open')
    )
    def toggle_export_modal(n1, n2, n3, is_open):
        """Open and close the Export Summary modal."""
        if n1 or n2 or n3:
            return not is_open
        return is_open

    # === Download summary data ===============================================
    @app.callback(
        [Output('export-description', 'hidden'),
         Output('export-confirm', 'hidden'),
         Output('export-modal-btns', 'hidden'),
         Output('export-close-btn', 'hidden')],
        Input('ok-export', 'n_clicks'),
        State('export-type', 'value'),
        State('memory-db', 'data')
    )
    def export_summary(n, file_type, data):
        """Export the SQA summary file and confirm the export."""
        if n == 0:
            raise PreventUpdate
        else:
            file = data['filename']
            data_type = data['type']
            default._export_sqa(file, data_type, file_type.lower())
            sleep(1.0)
            return [True, False, True, False]