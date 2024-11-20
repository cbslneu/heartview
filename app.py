from os import environ
from os import name as os_name
from dash import Dash, DiskcacheManager, CeleryManager
from heartview.dashboard.utils import _clear_temp, _clear_beat_editor_data, \
    _make_subdirs
from heartview.dashboard.layout import layout
from heartview.dashboard.callbacks import get_callbacks
import dash_bootstrap_components as dbc

sep = '\\' if os_name == 'nt' else '/'

if 'REDIS_URL' in environ:
    from celery import Celery
    celery_app = Celery(__name__,
                        broker = environ['REDIS_URL'],
                        backend = environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)
else:
    import diskcache
    cache = diskcache.Cache(f'.{sep}cache')
    background_callback_manager = DiskcacheManager(cache)

app = Dash(
    __name__,
    update_title = None,
    external_stylesheets = [dbc.themes.LUX, dbc.icons.FONT_AWESOME],
    background_callback_manager = background_callback_manager
)
app.title = 'PhysioView Dashboard'
app.layout = layout
app.server.name = 'PhysioView Dashboard'
get_callbacks(app)

if __name__ == '__main__':
    _make_subdirs()
    _clear_temp()
    _clear_beat_editor_data()
    app.run_server(debug = True)