import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

import pandas as pd


def import_dataframes():
    
    Assure = pd.read_csv('Clean Data/Assure.csv')
    Police = pd.read_csv('Clean Data/Police.csv')
    Vehicule = pd.read_csv('Clean Data/Vehicule.csv')
    train_df = pd.read_csv('Clean Data/train_df.csv')
    
    return Assure, Police, Vehicule, train_df

def create_timeseries():
    
    TimeSeries = pd.read_csv('Clean Data/MoisAccidents.csv')

    date = []
    m = 1
    
    for i in range(TimeSeries.shape[0]):
        date.append(str(m) + '-' + str(str(TimeSeries.iloc[i, -1])))
        if m < 12:
            m += 1
        else:
            m = 1
    
    TimeSeries['date'] = date
    
    return TimeSeries


def render_dashboard(server):
    dashboard_app = dash.Dash(server = server, routes_pathname_prefix = '/visualisation/', external_stylesheets = ['../static/css/style.css'])
    dashboard_app.title = 'Deployement - Dashboard'

    TimeSeries = create_timeseries()
    Assure, Police, Vehicule, train_df = import_dataframes()

    dashboard_app.layout = html.Div([
        
        html.Header([
            html.H1('Comité Générale des Assurance'),
            html.Span([
                html.A('Dashboard', href='/visualisation'),
                html.A('Tables', href='/navigation'),
                html.A('Prediction', href='/prediction'),
                html.A('Déconnexion', href='/', className = 'last'),
            ], className="menu")
        ]),
        
        html.Table([
            
            html.Tr([
                
                html.Td([
                    html.Div([
                        dcc.Graph(
                            id='gov-count',
                            figure=dict(
                                data=[
                                    dict(
                                        x=Assure['gouvernorat'].value_counts(
                                        ).index,
                                        y=Assure['gouvernorat'].value_counts(
                                        ).values,
                                        type='bar',
                                        marker=dict(
                                            color='#4775b4'
                                        )
                                    )
                                ],
                                layout=dict(
                                    title = '<b>Distribution Des Assurés Par Gouvernorat</b>',
                                    height = 400,
                                    width = 800,
                                    margin = dict(b = 150),
                                    font=dict(
                                        family = "Courier New, monospace",
                                        color = '#7f7f7f',
                                        size = 16,

                                    ),
                                )
                            )
                        )
                    ], className='pretty-container', style={'width': 800, 'height': 400}),
                ]),

                html.Td([
                    html.Div([
                        #html.H3('Répartition Des Accidents en Tunisie'),
                        html.Iframe(id='heatmap', srcDoc=open(
                            'Data Visualisation Workspace/map.html', 'r').read(), width='100%', height='100%')
                    ], className='pretty-container', style={'width': 300, 'height': 400})
                ])

            ]),

        ]),

        html.Table([

            html.Tr([

                html.Td([

                    html.Div([
                        
                        dcc.Dropdown(
                            id='pie-data-pick',
                            options=[
                                {'label': 'Type De Police', 'value': 'T'},
                                {'label': 'Nature De Police', 'value': 'N'}
                            ],
                            value='T',
                            className='picker'
                        ),
                        dcc.Graph(
                            id='police-count',
                            figure=go.Figure(
                                data=[
                                    go.Pie(
                                        labels=[
                                            'Individuel', 'Flotte'],
                                        values=Police['typePolice'].value_counts(
                                        ).values,
                                        hole=.5,
                                        marker_colors=[
                                            '#857aaa', '#389d63'],
                                    )
                                ],
                                layout=dict(
                                    title='<b>Répartition Des  Polices</b>',
                                    margin=dict(b=30, l=60),
                                    height=380,
                                    font=dict(
                                        family="Courier New, monospace",
                                        color='#7f7f7f',
                                        size=5,

                                    ),
                                )

                            )
                        ),
                    ], className='pretty-container', style={'width': 450, 'height': 450}),
                ]),

                html.Td([
                    html.Div([
                        dcc.Graph(
                            id='make-count',
                            figure=dict(
                                data=[
                                    dict(
                                        x=Vehicule['marque'].value_counts().head(15).index,
                                        y=Vehicule['marque'].value_counts().head(15).values,
                                        type='bar',
                                        marker=dict(
                                            color='#c39c4f'
                                        )
                                    )
                                ],
                                layout=dict(
                                    title='<b>Répartition des marques de véhicule</b>',
                                    margin=dict(b=140),
                                    font=dict(
                                        family="Courier New, monospace",
                                        color='#7f7f7f',
                                        size=16,
                                    ),
                                )
                            )
                        )
                    ], className='pretty-container', style={'width': 650, 'height': 450})
                ])

            ])

        ]),

        html.Table([
            html.Tr([
                html.Div([
                    dcc.Graph(
                        id='time-series2',
                        figure=dict(
                            data=[
                                dict(
                                    x=TimeSeries['date'],
                                    y=TimeSeries['accidentsChiffres'],
                                    marker=dict(
                                        color='#4775b4',
                                        symbol='diamond-open'
                                    ),
                                    mode='lines+markers',
                                    opacity=1
                                )
                            ],
                            layout=dict(
                                title='<b>Nombre d\'accidents par mois (2016 - 2019)</b>',
                                xaxis=dict(
                                    showgrid=False
                                ),
                                font=dict(
                                    family="Courier New, monospace",
                                    color='#7f7f7f',
                                    size=16,
                                ),
                            )
                        ),
                    ),
                ], className='pretty-container', style={'width': 1165, 'height': 450})
            ])
        ]),
    ], className='dashboard')

    @dashboard_app.callback(
        Output('police-count', 'figure'),
        [Input('pie-data-pick', 'value')]
    )
    def update_piechart(value):
        print(value)
        if value == 'T':
            return go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            'Individuel', 'Flotte'],
                        values=Police['typePolice'].value_counts(
                        ).values,
                        hole=.5,
                        marker_colors=['#857aaa', '#389d63'],
                    )
                ],
                layout=dict(
                    title='<b>Répartition Des Types De Polices</b>',
                    margin=dict(b=50, l=50),
                    transition={'duration': 500},
                    height=380,
                    font=dict(
                        family="Courier New, monospace",
                        color='#7f7f7f',
                        size=16,

                    ),
                )

            )
        elif value == 'N':
            print(Police['naturePolice'].value_counts())
            return go.Figure(
                data=[
                    go.Pie(
                        labels=['Renouvelable', 'Temporaire'],
                        values=Police['naturePolice'].value_counts(
                        ).values,
                        hole=.5,
                        marker_colors=['#857aaa', '#389d63'],
                    )
                ],
                layout=dict(
                    title='<b>Répartition De Polices</b>',
                    margin=dict(b=50, l=50),
                    transition={'duration': 500},
                    height=380,
                    font=dict(
                        family="Courier New, monospace",
                        color='#7f7f7f',
                        size=16,

                    ),
                )

            )
    

