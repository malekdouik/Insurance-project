import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State


def render_authentification(server):
    authentification_app = dash.Dash(server = server, routes_pathname_prefix = '/', external_stylesheets = ['../static/css/style.css'])
    authentification_app.title = 'Authentification'

    authentification_app.layout = html.Div([
        html.H1('Authentification', style={'textAlign': 'center', 'fontWeight': 'bold'}),
        html.Br(),
        html.Div([
            html.Center([
                dcc.Input(
                    id='login',
                    type='text',
                    placeholder='Login',
                    className='auth',
                ),
                html.Br(),
                dcc.Input(
                    id='password',
                    type='password',
                    placeholder='Mot de Passe',
                    className='auth',
                ),
                html.Br(),
                html.Button('Connexion', id = 'login-button', n_clicks=0, className='login-button'),
                html.Div(id='output', className='wrong-cred'),
                dcc.Location(id='location', refresh=True),
            ])
        ], style={'marginRight': 'auto', 'marginLeft': 'auto'})

    ], className = 'authentification-form')

    @authentification_app.callback(
        Output('location', 'href'),
        [Input('login-button', 'n_clicks')],
        [State('login', 'value'), State('password', 'value')]
    )
    def login(n_clicks, login, password):
        if (n_clicks > 0) & (login == 'root') & (password == 'root'):
            return '/visualisation'

    @authentification_app.callback(
        Output('output', 'children'),
        [Input('login-button', 'n_clicks')],
        [State('login', 'value'), State('password', 'value')]
    )
    def wrong_credentials(n_clicks, login, password):
        if ((login != 'root') | (password != 'root')) & n_clicks:
            return 'Wrong credentials'
        else: 
            return ''
    
