import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State

import datetime
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import io
import base64
import pickle

def import_dataframes():
    try:
        Assure = pd.read_csv('Clean Data/Assure.csv')
        Police = pd.read_csv('Clean Data/Police.csv')
        Vehicule = pd.read_csv('Clean Data/Vehicule.csv')
        BonusMalus = pd.read_csv('Clean Data/Vehicule.csv')
        Sinistre = pd.read_csv('Clean Data/Vehicule.csv')
        train_df = pd.read_csv('Clean Data/train_df.csv')
        fraude_df = pd.read_csv('Clean Data/FraudeTable_withResampling.csv')
    except:
        print('Could load dataframe(s) in dashboard.import_dataframes')
    else:
        return Assure, Police, Vehicule, BonusMalus, Sinistre, train_df, fraude_df


def render_prediction(server):
    prediction_app = dash.Dash(server = server, routes_pathname_prefix = '/prediction/', external_stylesheets = ['../static/css/style.css'])
    prediction_app.title = 'Deployment - Prediction'

    Assure, Police, Vehicule, BonusMalus, Sinistre, train_df, fraude_df = import_dataframes()

    prediction_app.layout = html.Div([
        html.Header([
            html.H1('Comité Générale des Assurance'),
            html.Span([
                html.A('Dashboard', href='/visualisation'),
                html.A('Tables', href='/navigation'),
                html.A('Prediction', href='/prediction'),
                html.A('Déconnexion', href='/', className='last'),
            ], className="menu")
        ]),
        html.Table([
            html.Tr([
                html.Td([
                    html.Div([
                        html.H2('Classement des clients'),
                        html.Center([
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Glisser et Déposer votre Fichier ou ',
                                    html.A('Sélectionner un Fichier', id='select-files', className='file-reader')
                                ], id='upload'),
                                style={
                                    'width': '80%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px',
                                    'color': 'grey'
                                },
                            ),
                        ]),
                        html.Br(),
                        html.Label('Classer un seul client ▼', className='drop-form'),
                        html.Br(),
                        
                        html.Table([
                            html.Tr([
                                html.Td([
                                    html.Label('Informations Compagnie', style={'color': '#3d9970', 'fontSize': '25px'}),
                                ]),
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Code Compagnie'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='code-compagnie',
                                        options=[{'label': codeC, 'value': codeC} for codeC in train_df['codeCompagnie'].unique()],
                                        clearable=False
                                    ),
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Code Agence'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='code-agence',
                                        options=[{'label': codeA, 'value': codeA} for codeA in Police['codeAgence'].unique()],
                                        clearable=False
                                    ),
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Informations Client', style={'color': '#3d9970', 'fontSize': '25px'}),
                                ]),
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Âge'),
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='age',
                                        min=18,
                                        max=110,
                                        value=20,
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Sexe'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='sexe',
                                        options=[
                                            {'label': 'Homme', 'value': 'H'},
                                            {'label': 'Femme', 'value': 'F'},
                                        ],
                                    ),
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Gouvernorat'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='gouvernorat',
                                        options=[{'label': gouvernorat, 'value': gouvernorat} for gouvernorat in train_df['gouvernorat'].unique()],
                                    ),
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Date d\'obtention du permis'),
                                ]),
                                html.Td([
                                    dcc.DatePickerSingle(
                                        id='date-obtention-permis',
                                        min_date_allowed=dt(1800, 1, 1),
                                        initial_visible_month=dt(2020, 1, 1),
                                        date=str(dt(2020, 1, 1))
                                    ),
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Informations Véhicule', style={ 'color': '#3d9970', 'fontSize': '25px'}),
                                ]),
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Marque de voiture'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='marque',
                                        options=[{'label': marque, 'value': marque} for marque in train_df['marque'].unique()],
                                    ),
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Energie'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='energie',
                                        options=[{'label': energie, 'value': energie} for energie in train_df['energie'].unique()],
                                    ),
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Puissance Fiscale'),
                                ]),
                            html.Td([
                                daq.NumericInput(
                                    id='puissance-fiscale',
                                    min=int(train_df['puissanceFiscal'].min()),
                                    max=int(train_df['puissanceFiscal'].max()),
                                    value=11
                                ),
                            ])
                        ]),
                        html.Tr([
                            html.Td([
                                html.Label('Usage du véhicule'),
                            ]),
                            html.Td([
                                dcc.Dropdown(
                                    id='usage',
                                    options=[{'label': usage, 'value': usage, 'title': usage} for usage in train_df['usage'].unique()],
                                    optionHeight=90,
                                ),
                            ])
                        ]),
                        html.Tr([
                            html.Td([
                                html.Label('Informations Police', style={ 'color': '#3d9970', 'fontSize': '25px'}),
                            ]),
                        ]),

                            html.Tr([
                                html.Td([
                                    html.Label('Type Police'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='type-police',
                                        options=[
                                            {'label': 'Individuel', 'value': 'I'},
                                            {'label': 'Flotte', 'value': 'F'}
                                        ],
                                    ),
                                ]),
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Nature Police'),
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='nature-police',
                                        options=[
                                            {'label': 'Temporaire', 'value': 'T'},
                                            {'label': 'Renouvelable', 'value': 'R'}
                                        ],
                                    ),
                                ]),
                            ]),
                            html.Tr([
                                html.Button('Evaluer >>', id='submit', n_clicks=0, className='submit')
                            ])
                        ])
                    ], className='form'),
                ]),
                html.Td([
                    html.Table([
                        html.Tr([html.Div([' '], id='prediction-output')]),
                        html.Tr([html.Div([' '], id='prediction-output2')])
                    ])
                ])
            ]),
            html.Tr([
                html.Td([
                    html.Div([
                        html.H2('Détection de Fraude'),
                        html.Br(),
                        html.Table([
                            html.Tr([
                                html.Td([
                                    html.Label('Classe Bonus-Malus')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='classeBM',
                                        min=int(fraude_df['classeBonusMalus'].min()),
                                        max=int(fraude_df['classeBonusMalus'].max()),
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Classe Bonus-Malus Compagnie')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='classeC',
                                        min=int(fraude_df['classeBonusMalusCompagnie'].min()),
                                        max=int(fraude_df['classeBonusMalusCompagnie'].max()),
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Classe Bonus-Malus CGA')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='classeCGA',
                                        min=int(fraude_df['classeBonusMalusCGA'].min()),
                                        max=int(fraude_df['classeBonusMalusCGA'].max()),
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Coefficient Bonus-Malus')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='coefBM',
                                        min=int(
                                            fraude_df['coefBonusMalus'].min()),
                                        max=int(
                                            fraude_df['coefBonusMalus'].max()),
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Coefficient Bonus-Malus Compagnie')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='coefC',
                                        min=int(fraude_df['coefBonusMalusCompagnie'].min()),
                                        max=int(fraude_df['coefBonusMalusCompagnie'].max()),
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Coefficient Bonus-Malus CGA')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='coefCGA',
                                        min=int(fraude_df['coefBonusMalusCGA'].min()),
                                        max=int(fraude_df['coefBonusMalusCGA'].max()),
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Etat du Contrat')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='etat-contrat',
                                        options=[{'label': 'En Vigueur', 'value': 1}, {'label': 'Terminé', 'value': 0}],
                                        clearable=False,
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Bonus')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='bonus',
                                        min=int(fraude_df['bonus'].min()),
                                        max=int(fraude_df['bonus'].max()),
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Dernière Classe Bonus-Malus')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='der-classeBM',
                                        min=int(fraude_df['dernierClassBonusMallus'].min()),
                                        max=int(fraude_df['dernierClassBonusMallus'].max()),
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Code Usage Véhicule')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='code-usage',
                                        options=[{'label': code, 'value': code} for code in fraude_df['codeUsage'].unique()],
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Code Compagnie')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='code-compagnie-fraude',
                                        options=[{'label': code, 'value': code} for code in fraude_df['codeCompagnie'].unique()],
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Code Agence')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='code-agence-fraude',
                                        options=[{'label': code, 'value': code} for code in fraude_df['codeAgence'].unique()],
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Type Intermédiaire')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='type-int',
                                        options=[{'label': type_int, 'value': type_int} for type_int in fraude_df['typeIntermediaire'].unique()],
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Nature Police')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='nature-police-fraude',
                                        options=[{'label': nature, 'value': nature} for nature in fraude_df['naturePolice'].unique()],
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Type Police')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='type-police-fraude',
                                        options=[{'label': type_police, 'value': type_police}
                                                 for type_police in fraude_df['typePolice'].unique()],
                                        
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Date Echéance Police')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='date-ech-police',
                                        min=int(fraude_df['dateEcheancePolice'].min()),
                                        max=int(fraude_df['dateEcheancePolice'].max()),
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Etat du Police')
                                ]),
                                html.Td([
                                    dcc.Dropdown(
                                        id='etat-police-fraude',
                                        options=[{'label': etat, 'value': etat} for etat in fraude_df['Etat_Police'].unique()]
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Jour Affectation')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='jour-aff',
                                        min=int(fraude_df['jour_Affectation'].min()),
                                        max=int(fraude_df['jour_Affectation'].max()),
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Jour Calcul')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='jour-cal',
                                        min=int(fraude_df['jour_Calcule'].min()),
                                        max=int(fraude_df['jour_Calcule'].max()),
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Jour Occurence')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='jour-occ',
                                        min=int(fraude_df['jour_Occurence'].min()),
                                        max=int(fraude_df['jour_Occurence'].max()),
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Td([
                                    html.Label('Jour Effet Police')
                                ]),
                                html.Td([
                                    daq.NumericInput(
                                        id='jour-eff-police',
                                        min=int(fraude_df['jour_EffetPolice'].min()),
                                        max=int(fraude_df['jour_EffetPolice'].max()),
                                    )
                                ])
                            ]),
                            html.Tr([
                                html.Button('Evaluer >>', id='submit-fraude', n_clicks=0, className='submit')
                            ])
                        ])
                    ], className='form')
                ]),
                html.Td([
                    html.Div(id='output-fraude')
                ])
            ])
        ]),
    ])

    @prediction_app.callback(
        Output('prediction-output2', 'children'),
        [Input('submit', 'n_clicks')],
        [State('code-compagnie', 'value'),
        State('code-agence', 'value'),
        State('age', 'value'),
        State('sexe', 'value'),
        State('gouvernorat', 'value'),
        State('date-obtention-permis', 'date'),
        State('marque', 'value'),
        State('energie', 'value'),
        State('puissance-fiscale', 'value'),
        State('usage', 'value'),
        State('type-police', 'value'),
        State('nature-police', 'value')]
    )
    def update_output2(n_clicks, codeC, codeA, age, sexe, gov, date, marque, energie, puissF, usage, tPolice, nPolice):
        if n_clicks:
            if None in [codeC, gov, date, marque, energie, puissF, usage]:
                print('Missing values')
            else:
                anneeExp = datetime.date.today().year - int(date[:4])
                enc_df = pd.DataFrame()
                cols = pd.read_csv('Workspace/train_df2.csv').columns[1:-1]
                for col in cols:
                    enc_df[col] = [0]
                enc_df['codeCompagnie_' + str(codeC)] = 1
                enc_df['gouvernorat_' + str(gov)] = 1
                enc_df['anneesExpConduite'] = anneeExp
                enc_df['marque_' + str(marque)] = 1
                enc_df['energie_' + str(energie)] = 1
                enc_df['puissanceFiscal'] = puissF
                enc_df['usage_' + str(usage)] = 1

                X = enc_df.values

                with open('Web App/models/class_client.pkl', 'rb') as file:
                    classifier = pickle.load(file)
                y = classifier.predict(X)
                
                return dcc.Graph(
                                id='meter-score',
                                figure=go.Figure().add_trace(go.Indicator(
                                    mode="number+delta",
                                    value=y[0],
                                    title={
                                        "text": "Classe Client",
                                    },
                                    delta={'reference': 0.5, 'relative': True},
                                )).update_layout(dict(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'),
                                ),
                                style=dict(
                                    width=500,
                                    height=500
                                ),
                            ),


    @prediction_app.callback(
        Output('prediction-output', 'children'),
        [Input('upload-data', 'contents'), 
        Input('upload-data', 'filename'),
        Input('upload-data', 'last_modified')]
    )
    def update_output(contents, filename, last_modified):
        attributes = ['CodeCompagnie', 'CodeAgence', 'Age', 'Sexe', 'Gouvernorat',
                      'DateObtentionPermis', 'MarqueVoiture', 'Energie', 'PuissanceFiscale',
                      'UsageVehicule', 'TypePolice', 'NaturePolice']
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
        except:
            return []
        else:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            if np.array([(att in df.columns) for att in attributes]).sum() < len(attributes):
                return ['Could not read file!']
            else:
                model_df = pd.DataFrame()

                model_df['codeCompagnie'] = df['CodeCompagnie'].values
                model_df['puissanceFiscal'] = df['PuissanceFiscale'].values
                model_df['energie'] = df['Energie'].values
                model_df['marque'] = df['MarqueVoiture'].values
                model_df['usage'] = df['UsageVehicule'].values
                model_df['gouvernorat'] = df['Gouvernorat'].values
                dates = df['DateObtentionPermis'].copy()
                for i in range(len(dates)):
                    dates[i] = datetime.date.today().year - int(str(dates[i])[:4])
                model_df['anneesExpConduite'] = dates

                model_df['codeCompagnie'] = model_df['codeCompagnie'].astype(str)
                categ = ['codeCompagnie', 'energie', 'marque', 'usage', 'gouvernorat']
                tmp = pd.get_dummies(model_df[categ])
                tmp['puissanceFiscal'] = model_df['puissanceFiscal'].values
                tmp['anneesExpConduite'] = model_df['anneesExpConduite'].values
                model_df = tmp

                cols = pd.read_csv('Workspace/train_df2.csv').columns[1:-1]
                enc_df = pd.DataFrame()
                for col in cols:
                    enc_df[col] = [0]*model_df.shape[0]
                for col in model_df.columns:
                    enc_df[col] = model_df[col]

                X = enc_df.values
                
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)
                
                with open('Web App/models/class_client.pkl', 'rb') as file:
                    classifier = pickle.load(file)
                y = classifier.predict(X)
                df['Classe'] = y
                df.to_csv('Web App/outputs/{}.csv'.format(filename.split('.')[0]))
                
                return html.Div([
                    dcc.Location(id='download', href='/download'),
                    dcc.Graph(
                        id='classes-output',
                        figure=go.Figure(
                            data=[
                                go.Pie(
                                    labels=[
                                        'Bon', 'Mauvais'],
                                    values=df['Classe'].value_counts(
                                    ).values,
                                    hole=.5,
                                    marker_colors=[
                                        '#857aaa', '#389d63'],
                                )
                            ],
                            layout=dict(
                                title='<b>Classes Des Assurés</b>',
                                height=400,
                                width=400,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(
                                    family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif",
                                    color='#7f7f7f',
                                    size=15,

                                ),
                            )

                        )
                    )
                ], className='class-output')
    
    @prediction_app.callback(
        Output('output-fraude', 'children'),
        [Input('submit-fraude', 'n_clicks')],
        [State('classeBM', 'value'), 
        State('classeC', 'value'),
        State('classeCGA', 'value'),
        State('coefBM', 'value'),
        State('coefC', 'value'),
        State('coefCGA', 'value'),
        State('etat-contrat', 'value'),
        State('bonus', 'value'),
        State('der-classeBM', 'value'),
        State('code-usage', 'value'),
        State('code-compagnie-fraude', 'value'),
        State('code-agence-fraude', 'value'),
        State('type-int', 'value'),
        State('nature-police-fraude', 'value'),
        State('type-police-fraude', 'value'),
        State('date-ech-police', 'value'),
        State('etat-police-fraude', 'value'),
        State('jour-aff', 'value'),
        State('jour-cal', 'value'),
        State('jour-occ', 'value'),
        State('jour-eff-police', 'value')]
    )
    def fraude(n_clicks, classeBM, classeC, classeCGA, coefBM, coefC, coefCGA, etatContrat, bonus, derClasseBM, codeUsage, codeC, codeA, typeInt, nPolice, tPolice, dateEchPolice, ePolice, jAff, jCal, jOcc, jEffPolice):
        if n_clicks:
            args = [classeBM, classeC, classeCGA, coefBM, coefC, coefCGA, etatContrat, bonus, 1, derClasseBM, codeUsage, codeC, codeA, typeInt, nPolice, tPolice, dateEchPolice, 1, ePolice, jAff, jCal, jOcc, jEffPolice]
            print(args)
            if None not in args:
                df = pd.DataFrame()
                i = 0
                for col in fraude_df.columns[1:]:
                    if col == 'Target':
                        continue
                    df[col] = [args[i]]
                    i += 1
                #df.drop(columns = ['Unnamed: 0'], inplace=True)
                print(df.head())
                with open('Web App/models/model.pkl', 'rb') as file:
                    classifier = pickle.load(file)
                
                y = classifier.predict(df.values)
                return dcc.Graph(
                    id='meter-score',
                    figure=go.Figure().add_trace(go.Indicator(
                        mode="number+delta",
                        value=y[0],
                        title={
                            "text": "Fraude",
                        },
                        delta={'reference': 0.5, 'relative': True},
                    )).update_layout(dict(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'),
                    ),
                    style=dict(
                        width=500,
                        height=500
                    ),
                ),
