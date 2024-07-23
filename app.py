from os import environ
from os import name as os_name
from dash import Dash, DiskcacheManager, CeleryManager
from heartview.dashboard.utils import _clear_temp
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
    cache = diskcache.Cache('./cache')
    background_callback_manager = DiskcacheManager(cache)

app = Dash(
    __name__,
    update_title = None,
    external_stylesheets = [dbc.themes.LUX, dbc.icons.FONT_AWESOME],
    background_callback_manager = background_callback_manager
)
app.title = 'HeartView Dashboard'
app.layout = layout
get_callbacks(app)

if __name__ == '__main__':
    _clear_temp()
    app.run_server()