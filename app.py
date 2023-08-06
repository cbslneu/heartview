from dash import Dash
from heartview.layout import layout
from heartview.callbacks import get_callbacks
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    update_title = None,
    external_stylesheets = [dbc.themes.LUX, dbc.icons.FONT_AWESOME]
)
app.title = 'HeartView Dashboard'
app.layout = layout
get_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug = True)