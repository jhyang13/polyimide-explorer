import base64
import io
import pathlib

import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

import pickle5 as pickle
from tensorflow.keras.models import Sequential, save_model, load_model

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

selected_keys_tg = pickle.load(open(DATA_PATH.joinpath("selected_keys_Tg.pickle"),"rb"))
df_tg = pd.DataFrame(columns = selected_keys_tg)

font_size_instruction = '13px'


modles = {}
for col in ['DensityValue', 'TensileModulusValue', 'TensileBreakValue', 'TgValue',
       'TdValue', 'TmValue', 'TensileYieldValue']:
    modles[col] = load_model(DATA_PATH.joinpath(col + '_Ensemble_TrainAllData.model'))

DF = pickle.load(open(DATA_PATH.joinpath("df_Plotly.pickle"),"rb"))
DF = DF.reset_index(drop=True)
df = DF.copy()

layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene={
            "camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 1.35, "y": -1.85, "z": 0.85},
            },
            "xaxis_title": "Young's Modulus E (GPa)",
            "yaxis_title": "Tensile Yield Strength σy (GPa)",
            "zaxis_title": "Glass Transition Temperature Tg (°C)"
        }
            
)


data = []

colorscale = [
    [0, "rgb(39, 143, 92)"],
    [0.3, "rgb(39, 143, 114)"],
    [0.4, "rgb(39, 143, 140)"],
    [0.5, "rgb(39, 115, 143)"],
    [0.65, "rgb(39, 84, 143)"],
    [1, "rgb(39, 56, 143)"],
]

# colorscale = [
#     [0, "rgb(244,236,21)"],
#     [0.3, "rgb(249,210,41)"],
#     [0.4, "rgb(134,191,118)"],
#     [0.5, "rgb(37,180,167)"],
#     [0.65, "rgb(17,123,215)"],
#     [1, "rgb(54,50,153)"],
# ]

scatter = go.Scatter3d(
    x=df["TensileModulusValue"],
    y=df["TensileYieldValue"],
    z=df["TgValue"],
    text=df["Smiles"],
    textposition="top center",
    mode="markers",
    marker=dict(size=df["MW"], 
                color=df["MW"], 
                colorscale = colorscale,
                colorbar = {"title": "Molecular<br>Weight"},
                sizeref = 90,
                line = {"color": "#444"},
                reversescale = True,
                sizemode = "diameter",
                opacity = 1,                
                symbol="circle"),
    hovertemplate='E: %{x:.2f} (GPa)<br>σy: %{y:.2f} (GPa)<br>Tg: %{z:.2f} (°C)',
    showlegend = False
)
data.append(scatter)

figure = go.Figure(data=data, layout=layout)



     
        

# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[

                    html.Div(
                        [
                            html.H4(
                                "Polyimide Explorer",
                                #"Discovery of Multi-Functional Polyimides through High-Throughput Screening using Explainable Machine Learning",
                                style={"font-weight": "bold", 'textAlign': 'center'}, 
                                #className="header_title",
                                #id="app-title",
                            ),
                            
                            html.Div( html.P(["This tool explores 8 million hypothetical polyimides for three thermal/mechanical properties including ", #html.Br(), 
                                             "Young's Modulus E, Tensile Yield Strength σy and Glass Transition Temperature Tg. All hypothetical ", #html.Br(), 
                                             "are obtained via the computational polycondensation of compounds that are commercially available. ", #html.Br(),
                                             "The best-performing 77,000 hypothetical polyimide are included for visualization. More details/data can be found in ", 
                                             html.A("Tao, Lei, Jinlong He, Vikas Varshney, Wei Chen, and Ying Li. 'Machine Learning Discovery of Multi-Functional Polyimides'", 
                                                    href=f"https://github.com/figotj/Polyimide_explorer", target="_blank")
                                           ]),
                                     
                                     ),                             

                        ],
                        #className="nine columns header_title_container",
                    ),

                    # html.Div(html.P(['Lei Tao', html.Sup('1'),
                    #                  ', Jinlong He', html.Sup('1'),
                    #                  ', Vikas Varshney', html.Sup('2'),
                    #                  ', Wei Chen', html.Sup('3'),
                    #                  ', Ying Li', html.Sup('1,4,*')],style={"font-weight": "bold", 'textAlign': 'center'})),
                    # html.Div(html.P([html.Sup('1'),'Department of Mechanical Engineering, University of Connecticut, Storrs, Connecticut 06269, United States', html.Br(), 
                    #                   html.Sup('2'),'Materials and Manufacturing Directorate, Air Force Research Laboratory, Wright-Patterson Air Force Base, Ohio 45433, United States', html.Br(),
                    #                   html.Sup('3'),'Department of Mechanical Engineering, Northwestern University, Evanston, Illinois 60208, United States', html.Br(),
                    #                   html.Sup('4'),'Polymer Program, Institute of Materials Science, University of Connecticut, Storrs, Connecticut 06269, United States', html.Br(),
                    #                   html.Sup('*'),'Corresponding author'],style={"font-size": 12, 'textAlign': 'center'})),
                   
                ],
            ),
            # Demo Description
            html.Div(
                className="row background",
                id="demo-explanation",
                style={"padding": "0px 0px"},
                children=[
                    
                    html.Div(
                        className="six columns",
                        children=[
                            
                            
                                html.Div(
                                    style={"margin-left": "15px"},
                                    children=[
                                                html.Span(
                                                    "Operation 1. ", style={'color': 'black', 'font-size': font_size_instruction}
                                                ),
                                                html.Span("Hover ", style={'font-weight': 'bold', 'font-style': 'italic', 'font-size': font_size_instruction}),
                                                html.Span(
                                                    "over a polyimide in the 3D plot to see its structure, and imide groups contained are highlighted",
                                                    style={'font-size': font_size_instruction}
                                                ),
                                                html.Br(),

                                                html.Span(
                                                    "Operation 3. ", style={'color': 'black', 'font-size': font_size_instruction}
                                                ),
                                                html.Span("Select ", style={'font-weight': 'bold', 'font-style': 'italic', 'font-size': font_size_instruction}),
                                                html.Span(
                                                    "the interested ranges of the three properties to filter desired hypothetical polyimides.",
                                                    style={'font-size': font_size_instruction}
                                                    
                                                ),
                                                html.Br(),

                                            ]
                                    )                            
                        
                            
                        ]),
                        
                    html.Div(
                        className="six columns",
                        children=[
                            
                                html.Div(
                                    style={"margin-left": "15px"},
                                    children=[
                                                html.Span(
                                                    "Operation 2. ", style={'color': 'black', 'font-size': font_size_instruction}
                                                ),                                                
                                                html.Span("Click ", style={'font-weight': 'bold', 'font-style': 'italic', 'font-size': font_size_instruction}),
                                                html.Span(
                                                    "a polyimide in the graph to see the info of its structure/property and reacting compounds at the bottom of the page.",
                                                    style={'font-size': font_size_instruction}
                                                ),
                                                html.Br(),

                                                
                                                html.Span(
                                                    "Operation 4. ", style={'color': 'black', 'font-size': font_size_instruction}
                                                ),
                                                html.Span("Predict ", style={'font-weight': 'bold', 'font-style': 'italic', 'font-size': font_size_instruction}),
                                                html.Span(
                                                    "the properties of polyimides based on a SMILES input.",
                                                    style={'font-size': font_size_instruction}
                                                    
                                                ),
                                            ]
                                    )                        
                            
                        ]),                    
                    
                    

                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            Card(
                                [
    
                                    html.Div(
                                        className="row background",
                                        id="space",
                                        style={"padding": "20px 20px"},
                                        children=[

                                                
                                        ],
                                    ),
                                    
                                    html.Div(
                                            style={"margin": "25px 5px 30px 0px"},
                                            children=[
                                                html.Div(
                                                    style={"margin-left": "5px"},
                                                    children=[

                                                        html.Div(id='rangeslider-E', 
                                                                 children=["Range of E: [0, 18] (GPa)"],
                                                                 style={'margin-top': 20}),
                                                        dcc.RangeSlider(id=f"rangeslider-x",
                                                                        min=0, 
                                                                        max=18, 
                                                                        marks={
                                                                            i: str(i) for i in [0, 6, 12, 18]
                                                                        },
                                                                        value=[0, 18],         
                                                                        dots=False,
                                                                        step=0.01,
                                                                        updatemode='mouseup')
                                                    ],
                                                ),
                                            ],
                                    ),
                                    html.Div(
                                        children=[
                                                html.Div(
                                                    style={"margin-left": "5px"},
                                                    children=[
                                                        html.Div(id='rangeslider-Y', 
                                                                 children=["Range of σy: [0, 0.3] (GPa)"],
                                                                 style={'margin-top': 20}),
                                                        dcc.RangeSlider(id='rangeslider-y',
                                                                        min=0, 
                                                                        max=0.3, 
                                                                        marks={i: str(i) for i in [0, 0.1, 0.2, 0.3]},
                                                                        value=[0, 0.3],         
                                                                        dots=False,
                                                                        step=0.01,
                                                                        updatemode='mouseup')
                                                    ],
                                                ),
                                            ],
                                    ),
                                    html.Div(
                                            style={"margin": "25px 5px 30px 0px"},
                                            children=[
                                                html.Div(
                                                    style={"margin-left": "5px"},
                                                    children=[

                                                        html.Div(id='rangeslider-Tg', 
                                                                 children=["Range of Tg: [0, 550] (°C)"],
                                                                 style={'margin-top': 20}),
                                                        dcc.RangeSlider(id=f"rangeslider-z",
                                                                        min=0, 
                                                                        max=550, 
                                                                        marks={i: str(i) for i in [0, 100, 200, 300, 400, 500, 550]},
                                                                        value=[0, 550],         
                                                                        dots=False,
                                                                        step=0.01,
                                                                        updatemode='mouseup')
                                                    ],
                                                ),
                                            ],
                                    ),
                                    html.Div(
                                        [
                                            html.Div(id="chem_img"),
                                            html.Div(id="chem_name"),
                                            html.Div(
                                                id="chem_desc",
                                                children=[dcc.Markdown([])],
                                                style={
                                                    "text-align": "center",
                                                    "margin-bottom": "7px",
                                                    "font-weight": "bold",
                                                },
                                            ),
                                        ],
                                        className="chem__desc__container",
                                    ),

    
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", figure=figure, style={"height": "70vh"})
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
                        style={'border': '0px green solid',"margin-top": "100px"},
                        children=[
                                    html.Div([
                                        html.Div(dcc.Input(id='to_be_pridicted', placeholder="Input box for SMILES", type='text',
                                                           style={'width': '80%',"text-align": "center",'border': '1px black solid','background-color': '#f2f3f4'})),
                                        html.Div(
                                            html.Button(id="predict", children=["Predict"],style={'width': '80%','border': '0.5px black solid'})
                                        ),
                                        html.Div(id='button_result_1'),
                                        html.Div(id='predict_img', style={'border': '0px green solid'}),
                                        html.Div(id='button_result_2'),
                                    ])
                                

                                ],

                    ),
                ],
            ),

            html.Div(
                className="row background",
                id="table-explanation",
                style={"padding": "50px 45px"},
                children=[
                        html.Table([
                                        html.Tr([html.Td(['Smiles'],style={"font-weight": "bold"}), html.Td(id='Smiles')]),
                                        
                                        # html.Tr([html.Td(['Reaction'],style={"font-weight": "bold"}), 
                                        #          html.Td([
                                        #              html.Span(id='A_img', style={'verticalAlign': 'center'}), 
                                        #              html.Span(['+'], style={'verticalAlign': 'center'}), 
                                        #              html.Span(id='B_img', style={'verticalAlign': 'center'}), 
                                        #              html.Span(['='], style={'verticalAlign': 'center'}), 
                                        #              html.Span(id='Smiles_img', style={'verticalAlign': 'center'})
                                        #          ])
                                        # ]),
                                        html.Tr([html.Td(['E'],style={"font-weight": "bold"}), html.Td(id='TensileModulusValue')]),
                                        html.Tr([html.Td(['σ', html.Sub('y')],style={"font-weight": "bold"}), html.Td(id='TensileYieldValue')]),
                                        html.Tr([html.Td(['Tg'],style={"font-weight": "bold"}), html.Td(id='TgValue')]),                                        
                                        html.Tr([html.Td(['Reaction route'],style={"font-weight": "bold"}), 
                                                 html.Td([
                                                        html.Div([
                                                            html.Div(id='A_img', style={'display': 'inline-block','verticalAlign': 'middle','border': '0px green solid'}),
                                                            html.Div([html.Div(id='plus_sign', style={'color': 'black','font-size': '50px'})]
                                                                     ,style={'display': 'inline-block','verticalAlign': 'middle','border': '0px red solid'}),
                                                            html.Div(id='B_img', style={'display': 'inline-block','verticalAlign': 'middle','border': '0px green solid'}),
                                                            html.Div([html.Div(id='equal_sign', style={'color': 'black','font-size': '50px'})]
                                                                     ,style={'display': 'inline-block','verticalAlign': 'middle','border': '0px red solid'}),
                                                            html.Div(id='Smiles_img', style={'display': 'inline-block','verticalAlign': 'middle','border': '0px green solid'}),
                                                        ], style={'display': 'block','height': '200px'}),
                                                 ])
                                        ]),

                                        html.Tr([html.Td(['Component A'],style={"font-weight": "bold"}), html.Td(id='Component_A')]),
                                        html.Tr([html.Td(['Smiles of A'],style={"font-weight": "bold"}), html.Td(id='Smiles_A')]),
                                        html.Tr([html.Td(['PubChem CID of A'],style={"font-weight": "bold"}), html.Td(id='CID_A')]),
                                        html.Tr([html.Td(['Component B'],style={"font-weight": "bold"}), html.Td(id='Component_B')]),
                                        html.Tr([html.Td(['Smiles of B'],style={"font-weight": "bold"}), html.Td(id='Smiles_B')]),
                                        html.Tr([html.Td(['PubChem CID of B'],style={"font-weight": "bold"}), html.Td(id='CID_B')]),
                                    ]),
                ],
            ),
        ],
    )


def demo_callbacks(app):

    def df_row_from_hover(hoverData):
        """ Returns row for hover point as a Pandas Series. """
    
        try:
            point_number = hoverData["points"][0]["pointNumber"]
            molecule_name = str(figure["data"][0]["text"][point_number]).strip()
            return df.loc[df["Smiles"] == molecule_name]
        except KeyError as error:
            print(error)
            return pd.Series()

    @app.callback(
        Output('graph-3d-plot-tsne', 'figure'),
        [Input('rangeslider-x', 'value'),
         Input('rangeslider-y', 'value'),
         Input('rangeslider-z', 'value')])
    def update_output(x, y, z):
        Flag =  (DF['TensileModulusValue'] >= x[0]) & \
                (DF['TensileModulusValue'] <= x[1]) & \
                (DF['TensileYieldValue'] >= y[0]) & \
                (DF['TensileYieldValue'] <= y[1]) & \
                (DF['TgValue'] >= z[0]) & \
                (DF['TgValue'] <= z[1])
        df = DF[Flag]
        

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene={
                    "camera": {
                        "up": {"x": 0, "y": 0, "z": 1},
                        "center": {"x": 0, "y": 0, "z": 0},
                        "eye": {"x": 1.35, "y": -1.85, "z": 0.85},
                    },
                    "xaxis":{"range": [0,18]},
                    "yaxis":{"range": [0,0.3]},
                    "zaxis":{"range": [0,550]},
                    "xaxis_title": "Young's Modulus E (GPa)",
                    "yaxis_title": "Tensile Yield Strength σy (GPa)",
                    "zaxis_title": "Glass Transition Temperature Tg (°C)"
                }
                    
        )
        
        
        data = []
            
        scatter = go.Scatter3d(
            x=df["TensileModulusValue"],
            y=df["TensileYieldValue"],
            z=df["TgValue"],
            text=df["Smiles"],
            textposition="top center",
            mode="markers",
            marker=dict(size=df["MW"], 
                        color=df["MW"], 
                        colorscale = colorscale,
                        colorbar = {"title": "Molecular Weight"},
                        sizeref = 90,
                        line = {"color": "#444"},
                        reversescale = True,
                        sizemode = "diameter",
                        opacity = 1,                
                        symbol="circle"),
            hovertemplate='E: %{x:.2f} (GPa)<br>σy: %{y:.2f} (GPa)<br>Tg: %{z:.2f} (°C)',
            showlegend = False
        )
        data.append(scatter)
        
        figure = go.Figure(data=data, layout=layout)        
        

        return figure


    @app.callback(
        [
            Output('button_result_1', 'children'),
            Output('predict_img', 'children'),
            Output('button_result_2', 'children'),
            
        ],
        [Input("predict", "n_clicks")],
        [State('to_be_pridicted', 'value')],
    )
    def update_output(n_clicks, value):
        if value is None:
            return [html.Div('Enter a polyimide SMILES and press predict for property prediction'), None, None]
        else:
           df_query = pd.DataFrame([value])
           molecules = df_query[0].apply(Chem.MolFromSmiles)
        
           df_query.loc[:,['molecules']] = molecules
           df_query = df_query.dropna()
           if len(df_query) == 0:
                return [html.Div('Not a valid SMILES'), None, None]
           else:
                smiles = df_query[0].iloc[0]
                buffered = BytesIO()
                size = 300
                d2d = rdMolDraw2D.MolDraw2DSVG(size, size)
                opts = d2d.drawOptions()
                opts.clearBackground = False
                
                opts.bondLineWidth = 1
                d2d.drawOptions().setHighlightColour((0.5,0.0,0.5))
                mol = Chem.MolFromSmiles(smiles)
                patt = Chem.MolFromSmarts('C(=O)NC(=O)')
                
                if len(mol.GetSubstructMatches(patt)) == 0:
                    d2d.DrawMolecule(mol)
                    d2d.FinishDrawing()
                    img_str = d2d.GetDrawingText()
                    buffered.write(str.encode(img_str))
                    img_str = base64.b64encode(buffered.getvalue())
                    img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
                    
                    Message_1 = html.Div('Warning: No imide group detected, not a valid polyimide:',style={'color': 'red','font-weight': 'bold'})
                    Smiles_img = html.Img(
                            src=img_str,
                            style={
                                "width": "300px",
                                "background-color": f"rgba(255,255,255,{0})",
                            })                    
                else:
                    Hit_ats = []
                    Hit_bonds = []
                    hit_ats = list(mol.GetSubstructMatches(patt)[0])
                    hit_bonds = []
                    for bond in patt.GetBonds():
                        aid1 = hit_ats[bond.GetBeginAtomIdx()]
                        aid2 = hit_ats[bond.GetEndAtomIdx()]
                        hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
                        Hit_ats = Hit_ats + hit_ats
                        Hit_bonds = Hit_bonds + hit_bonds
                        
                    if len(mol.GetSubstructMatches(patt)) == 2:
                        hit_ats = list(mol.GetSubstructMatches(patt)[1])
                        hit_bonds = []
                        for bond in patt.GetBonds():
                            aid1 = hit_ats[bond.GetBeginAtomIdx()]
                            aid2 = hit_ats[bond.GetEndAtomIdx()]
                            hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
                            Hit_ats = Hit_ats + hit_ats
                            Hit_bonds = Hit_bonds + hit_bonds
                
                    rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=Hit_ats,
                                                        highlightBonds=Hit_bonds)
                    d2d.DrawMolecule(mol)
                    
                    #d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
                    d2d.FinishDrawing()
                    img_str = d2d.GetDrawingText()
                    buffered.write(str.encode(img_str))
                    img_str = base64.b64encode(buffered.getvalue())
                    img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
                    
                    Message_1 = html.Div('Accepted: Imide group detected, a valid polyimide:',style={'color': 'green','font-weight': 'bold'})
                    Smiles_img = html.Img(
                            src=img_str,
                            style={
                                "width": "300px",
                                "background-color": f"rgba(255,255,255,{0})",
                            })
            
                fp = df_query.molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
                fp_n = fp.apply(lambda m: m.GetNonzeroElements())
            
                MY_finger_tg = pd.DataFrame()
            
                for polymer in fp_n:
                    df_row = pd.DataFrame(polymer, index=[0])
                    result = pd.concat([df_tg, df_row])
                    result = result[selected_keys_tg]
                    MY_finger_tg = MY_finger_tg.append(result, ignore_index=True)
            
                X_tg = MY_finger_tg.fillna(0)
            
                ML_results = pd.DataFrame()
            
                for col in ['DensityValue', 'TensileModulusValue', 'TensileBreakValue', 'TgValue',
                       'TdValue', 'TmValue', 'TensileYieldValue']:
                    ML_results[col] = [i[0] for i in modles[col].predict((X_tg))]
                    #print(col, ML_results[col])
                Message_2 = html.Div(
                                    [
                                        html.Span(
                                               "Predicted E = {:0.2f} (GPa)".format(
                                                ML_results['TensileModulusValue'].iloc[0],
                                                )
                                        ),
                                        html.Br(),
            
                                        html.Span(
                                               "Predicted σy = {:0.2f} (GPa)".format(
                                                ML_results['TensileYieldValue'].iloc[0],
                                                )
                                        ),
                                        html.Br(),
                                        
                                        html.Span(
                                               "Predicted Tg = {:0.2f} (°C)".format(
                                                ML_results['TgValue'].iloc[0],
                                                )
                                        ),
                                        html.Br(),    
                                    ]
                                )      
                return [Message_1,Smiles_img,Message_2]
        
    
    @app.callback(
        Output('rangeslider-E', 'children'),
        [Input('rangeslider-x', 'value')])
    def update_output(value):
        return "Range of E: [{:0.2f}, {:0.2f}] (GPa)".format(
            value[0],
            value[1]
    )
    
    @app.callback(
        Output('rangeslider-Y', 'children'),
        [Input('rangeslider-y', 'value')])
    def update_output(value):
        return "Range of σy: [{:0.2f}, {:0.2f}] (GPa)".format(
            value[0],
            value[1]
    )
    
    @app.callback(
        Output('rangeslider-Tg', 'children'),
        [Input('rangeslider-z', 'value')])
    def update_output(value):
        return "Range of Tg: [{:0.2f}, {:0.2f}] (°C)".format(
            value[0],
            value[1]
    )    
    @app.callback(
        [
            Output('Smiles', 'children'),
            Output("Smiles_img", "children"),
            Output("A_img", "children"),
            Output("B_img", "children"),
            Output('TensileModulusValue', 'children'),
            Output('TensileYieldValue', 'children'),
            Output('TgValue', 'children'),
            Output('Component_A', 'children'),
            Output('Smiles_A', 'children'),
            #Output("A_img", "children"),
            Output('CID_A', 'children'),
            Output('Component_B', 'children'),
            Output('Smiles_B', 'children'),
            #Output("B_img", "children"),
            Output('CID_B', 'children'),
            Output('plus_sign', 'children'),
            Output('equal_sign', 'children'),
        ],
        [Input("graph-3d-plot-tsne", "clickData")],)
    def callback_a(clickData):
        
        if clickData is None:
            raise PreventUpdate

        try:
            df_row = df_row_from_hover(clickData)
            if df_row.empty:
                raise Exception
            
            size = 300
            #Polymer Smiles to Img
            smiles = df_row['Smiles'].iloc[0]
            buffered = BytesIO()
            d2d = rdMolDraw2D.MolDraw2DSVG(size, size)
            opts = d2d.drawOptions()
            opts.clearBackground = False
            
            opts.bondLineWidth = 1
            d2d.drawOptions().setHighlightColour((0.5,0.0,0.5))
            mol = Chem.MolFromSmiles(smiles)
            patt = Chem.MolFromSmarts('C(=O)NC(=O)')
            Hit_ats = []
            Hit_bonds = []
            hit_ats = list(mol.GetSubstructMatches(patt)[0])
            hit_bonds = []
            for bond in patt.GetBonds():
                aid1 = hit_ats[bond.GetBeginAtomIdx()]
                aid2 = hit_ats[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
                Hit_ats = Hit_ats + hit_ats
                Hit_bonds = Hit_bonds + hit_bonds
                
            if len(mol.GetSubstructMatches(patt)) == 2:
                hit_ats = list(mol.GetSubstructMatches(patt)[1])
                hit_bonds = []
                for bond in patt.GetBonds():
                    aid1 = hit_ats[bond.GetBeginAtomIdx()]
                    aid2 = hit_ats[bond.GetEndAtomIdx()]
                    hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
                    Hit_ats = Hit_ats + hit_ats
                    Hit_bonds = Hit_bonds + hit_bonds
        
            rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=Hit_ats,
                                                highlightBonds=Hit_bonds)
            d2d.DrawMolecule(mol)
            
            #d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
            d2d.FinishDrawing()
            img_str = d2d.GetDrawingText()
            buffered.write(str.encode(img_str))
            img_str = base64.b64encode(buffered.getvalue())
            img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
                
            Smiles_img = html.Img(
                    src=img_str,
                    style={
                        "width": "200px",
                        "background-color": f"rgba(255,255,255,{0})",
                    })

            #Component A Smiles to Img
            smiles = df_row['Smi_A'].iloc[0]
            buffered = BytesIO()
            d2d = rdMolDraw2D.MolDraw2DSVG(size, size)
            opts = d2d.drawOptions()
            opts.clearBackground = False
            
            opts.bondLineWidth = 1
            # d2d.drawOptions().setHighlightColour((0.5,0.0,0.5))
            # mol = Chem.MolFromSmiles(smiles)
            # patt = Chem.MolFromSmarts('[NH2]')
            # # if len(mol.GetSubstructMatches(patt)) == 0:
            # #     patt = Chem.MolFromSmarts('N=C=O')
            # # else:
            # #     patt = Chem.MolFromSmarts('[NH2]')
            # Hit_ats = []
            # Hit_bonds = []
            # hit_ats = list(mol.GetSubstructMatches(patt)[0])
            # hit_bonds = []
            # for bond in patt.GetBonds():
            #     aid1 = hit_ats[bond.GetBeginAtomIdx()]
            #     aid2 = hit_ats[bond.GetEndAtomIdx()]
            #     hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
            #     Hit_ats = Hit_ats + hit_ats
            #     Hit_bonds = Hit_bonds + hit_bonds
                
            # if len(mol.GetSubstructMatches(patt)) == 2:
            #     hit_ats = list(mol.GetSubstructMatches(patt)[1])
            #     hit_bonds = []
            #     for bond in patt.GetBonds():
            #         aid1 = hit_ats[bond.GetBeginAtomIdx()]
            #         aid2 = hit_ats[bond.GetEndAtomIdx()]
            #         hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
            #         Hit_ats = Hit_ats + hit_ats
            #         Hit_bonds = Hit_bonds + hit_bonds
        
            # rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=Hit_ats,
            #                                     highlightBonds=Hit_bonds)
            # d2d.DrawMolecule(mol)
            
            d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
            d2d.FinishDrawing()
            img_str = d2d.GetDrawingText()
            buffered.write(str.encode(img_str))
            img_str = base64.b64encode(buffered.getvalue())
            img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
                
            A_img = html.Img(
                    src=img_str,
                    style={
                        "width": "150px",
                        "background-color": f"rgba(255,255,255,{0})",
                    })            

            #Component B Smiles to Img
            smiles = df_row['Smi_Dianhydride'].iloc[0]
            buffered = BytesIO()
            d2d = rdMolDraw2D.MolDraw2DSVG(size, size)
            opts = d2d.drawOptions()
            opts.clearBackground = False
            
            opts.bondLineWidth = 1
            # d2d.drawOptions().setHighlightColour((0.5,0.0,0.5))
            # mol = Chem.MolFromSmiles(smiles)
            # patt = Chem.MolFromSmarts('C(=O)OC(=O)')
            # Hit_ats = []
            # Hit_bonds = []
            # hit_ats = list(mol.GetSubstructMatches(patt)[0])
            # hit_bonds = []
            # for bond in patt.GetBonds():
            #     aid1 = hit_ats[bond.GetBeginAtomIdx()]
            #     aid2 = hit_ats[bond.GetEndAtomIdx()]
            #     hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
            #     Hit_ats = Hit_ats + hit_ats
            #     Hit_bonds = Hit_bonds + hit_bonds
                
            # if len(mol.GetSubstructMatches(patt)) == 2:
            #     hit_ats = list(mol.GetSubstructMatches(patt)[1])
            #     hit_bonds = []
            #     for bond in patt.GetBonds():
            #         aid1 = hit_ats[bond.GetBeginAtomIdx()]
            #         aid2 = hit_ats[bond.GetEndAtomIdx()]
            #         hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
            #         Hit_ats = Hit_ats + hit_ats
            #         Hit_bonds = Hit_bonds + hit_bonds
        
            # rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=Hit_ats,
            #                                     highlightBonds=Hit_bonds)
            # d2d.DrawMolecule(mol)
            
            d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
            d2d.FinishDrawing()
            img_str = d2d.GetDrawingText()
            buffered.write(str.encode(img_str))
            img_str = base64.b64encode(buffered.getvalue())
            img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
                
            B_img = html.Img(
                    src=img_str,
                    style={
                        "width": "150px",
                        "background-color": f"rgba(255,255,255,{0})",
                    })
            #print (df_row)
            return (df_row['Smiles'].iloc[0], 
                    Smiles_img,
                    A_img,
                    B_img,
                    '{:0.2f} (GPa)'.format(df_row['TensileModulusValue'].iloc[0]), 
                    '{:0.2f} (GPa)'.format(df_row['TensileYieldValue'].iloc[0]), 
                    '{:0.2f} (°C)'.format(df_row['TgValue'].iloc[0]), 
                    df_row['Comp_A'].iloc[0], 
                    df_row['Smi_A'].iloc[0], 
                    #None,#A_img,
                    html.A(str(df_row['CID_A'].iloc[0]) + " (click to link to PubChem database for the detail of the component)", href=f"https://pubchem.ncbi.nlm.nih.gov/compound/{df_row['CID_A'].iloc[0]}", target="_blank"),
                    'SA_Dianhydride', 
                    df_row['Smi_Dianhydride'].iloc[0],
                    #None,#B_img,
                    html.A(str(df_row['CID_Dianhydride'].iloc[0]) + " (click to link to PubChem database for the detail of the component)", href=f"https://pubchem.ncbi.nlm.nih.gov/compound/{df_row['CID_Dianhydride'].iloc[0]}", target="_blank"),
                    '+',
                    '='
                    )


    
        except Exception as error:
            print(error)
            raise PreventUpdate
            
    @app.callback(
        [

            Output("chem_img", "children"),

        ],
        [Input("graph-3d-plot-tsne", "hoverData")],
    )
    def chem_info_on_hover(hoverData):
        """
        Display chemical information on graph hover.
        Update the image, link, description.
    
        :params hoverData: data on graph hover
        """
    
        if hoverData is None:
            raise PreventUpdate
    
        try:
            df_row = df_row_from_hover(hoverData)
            if df_row.empty:
                raise Exception
                
    
            smiles = df_row['Smiles'].iloc[0]
            buffered = BytesIO()
            d2d = rdMolDraw2D.MolDraw2DSVG(300, 300)
            opts = d2d.drawOptions()
            opts.clearBackground = False
            
            opts.bondLineWidth = 1
            d2d.drawOptions().setHighlightColour((0.5,0.0,0.5))
            mol = Chem.MolFromSmiles(smiles)
            patt = Chem.MolFromSmarts('C(=O)NC(=O)')
            Hit_ats = []
            Hit_bonds = []
            hit_ats = list(mol.GetSubstructMatches(patt)[0])
            hit_bonds = []
            for bond in patt.GetBonds():
                aid1 = hit_ats[bond.GetBeginAtomIdx()]
                aid2 = hit_ats[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
                Hit_ats = Hit_ats + hit_ats
                Hit_bonds = Hit_bonds + hit_bonds
                
            if len(mol.GetSubstructMatches(patt)) == 2:
                hit_ats = list(mol.GetSubstructMatches(patt)[1])
                hit_bonds = []
                for bond in patt.GetBonds():
                    aid1 = hit_ats[bond.GetBeginAtomIdx()]
                    aid2 = hit_ats[bond.GetEndAtomIdx()]
                    hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
                    Hit_ats = Hit_ats + hit_ats
                    Hit_bonds = Hit_bonds + hit_bonds
        
            rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=Hit_ats,
                                                highlightBonds=Hit_bonds)
            d2d.DrawMolecule(mol)

    
            #d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
            d2d.FinishDrawing()
            img_str = d2d.GetDrawingText()
            buffered.write(str.encode(img_str))
            img_str = base64.b64encode(buffered.getvalue())
            img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
            # img_str = df_data.query(f"{col} == @smiles")[f"{col}_img"].values[0]
            
    
            image = html.Img(
                    src=img_str,
                    style={
                        "width": "100%",
                        "background-color": f"rgba(255,255,255,{0})",
                    })
        
            desc = html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(smiles)],
                    )
               
            return (
                image,
            )
    
        except Exception as error:
            print(error)
            raise PreventUpdate
