import dash
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd

Vehicule = pd.read_csv('Clean Data/Vehicule.csv')

def reformat(s):
    for i in range(len(s)):
        if s[i].isupper():
            s = s[:i] + ' ' + s[i:]
            break 
    if len(s[11:]) > 0:
        s = s[:11] + '...'
    return s.upper()

def generate_table(df, table_name):
    return dash_table.DataTable(
        id='data-table',

        columns=[{'name': [table_name, reformat(i)], 'id': i} for i in df.columns],
        data=df.head(100).to_dict('records'),

        page_size=10,
        merge_duplicate_headers=True,

        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,
            'textAlign': 'center',
            'fontFamily': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif",
        },

        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,
        },

        style_as_list_view=True,
    ),

def import_dataframes():
    
    Assure = pd.read_csv('Clean Data/Assure.csv')
    Police = pd.read_csv('Clean Data/Police.csv')
    Vehicule = pd.read_csv('Clean Data/Vehicule.csv')
    BonusMalus = pd.read_csv('Clean Data/BonusMalus.csv')
    Sinistre = pd.read_csv('Clean Data/Sinistre.csv')
    
    return Assure, Police, Vehicule, BonusMalus, Sinistre

def render_navigation(server):
    navigation_app = dash.Dash(server = server, routes_pathname_prefix = '/navigation/', external_stylesheets = ['../static/css/style.css'])
    navigation_app.title = 'Deployment - Tables'

    Assure, Police, Vehicule, BonusMalus, Sinistre = import_dataframes()
    
    navigation_app.layout = html.Div([
        
        html.Header([
            html.H1('Comité Générale des Assurance'),
            html.Span([
                html.A('Dashboard', href='/visualisation'),
                html.A('Tables', href='/navigation'),
                html.A('Prediction', href='/prediction'),
                html.A('Déconnexion', href='/', className='last'),
            ], className="menu")
        ]),

        html.Div([
            html.Table([
                html.Tr([
                    html.Td([
                        html.Button('VEHICULE', id='button-vehicule', className = 'datatable-menu'),
                        html.Br(),
                        html.Button('ASSURE', id='button-assure', className = 'datatable-menu'),
                        html.Br(),
                        html.Button('POLICE', id='button-police', className = 'datatable-menu'),
                        html.Br(),
                        html.Button('BONUS-MALUS', id='button-bonusmalus', className = 'datatable-menu'),
                        html.Br(),
                        html.Button('SINISTRE', id='button-sinistre', className = 'datatable-menu'),
                    ], style={'width': '5%'}),
                    html.Td([], id='table-display-column')
                ])
            ])
        ], className='table-display')
    ])

    @navigation_app.callback(
        Output('table-display-column', 'children'),
        [Input('button-vehicule', 'n_clicks'),
        Input('button-assure', 'n_clicks'),
        Input('button-police', 'n_clicks'),
        Input('button-bonusmalus', 'n_clicks'),
        Input('button-sinistre', 'n_clicks')]
    )
    def update_table(vehicule, assure, police, bonusmalus, sinistre):
        ctx = dash.callback_context
        clicked = ctx.triggered[0]['prop_id'].split('.')[0]

        if (vehicule is None) & (assure is None) & (police is None) & (bonusmalus is None) & (sinistre is None):
            return generate_table(Vehicule, 'VEHICULE')

        if clicked == 'button-vehicule':
            return generate_table(Vehicule, 'VEHICULE')
        elif clicked == 'button-assure':
            return generate_table(Assure, 'ASSURE')
        elif clicked == 'button-police':
            return generate_table(Police, 'POLICE')
        elif clicked == 'button-bonusmalus':
            return generate_table(BonusMalus, 'BONUS-MALUS')
        elif clicked == 'button-sinistre':
            return generate_table(Sinistre, 'SINISTRE')
