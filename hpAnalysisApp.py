import pandas as pd
import os
import numpy as np
import gunicorn
from plotly.tools import mpl_to_plotly
import plotly.express as px
import plotly.graph_objs as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import difflib
import dash_core_components as dcc
import dash
#from dash import dcc
#import dash_html_components as html
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from collections import Counter
#import dash_table
import matplotlib.pyplot as plt
#from data_model import hp_df_final



pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)


#movies_df = pd.read_csv('C:/Users/Willie/PycharmProjects/hpDashboard/Chapters.csv', encoding = "ISO-8859-1", engine='python')

#os.chdir('C:/Users/Willie/PycharmProjects/hpApplication')
#os.chdir('C:/Users/Willie/PycharmProjects/HarryPotterApp')

'''
# current directory csv files
csvs = [x for x in os.listdir('.') if x.endswith('.csv')]

fns = [os.path.splitext(os.path.basename(x))[0] for x in csvs]

#print(fns)

# ingest csv's as dictionary items
dfs = {}
for i in range(len(fns)):
    dfs[fns[i]] = pd.read_csv(csvs[i], encoding = "ISO-8859-1", engine='python')

# create df from dictionary items
chapter_df = dfs['Chapters']
characters_df = dfs['Characters']
dialogue_df = dfs['Dialogue']
movies_df = dfs['Movies']

#Movie ID key reads in incorrectyl in pycharm
movies_df=movies_df.rename(columns = {'ï»¿Movie ID':'Movie ID'})


places_df = dfs['Places']
spells_df = dfs['Spells']
'''

chapter_df = pd.read_csv('Chapters.csv', encoding = "ISO-8859-1", engine='python')
characters_df = pd.read_csv('Characters.csv', encoding = "ISO-8859-1", engine='python')
dialogue_df = pd.read_csv('Dialogue.csv', encoding = "ISO-8859-1", engine='python')

movies_df = pd.read_csv('Movies.csv', encoding = "ISO-8859-1", engine='python')
movies_df=movies_df.rename(columns = {'ï»¿Movie ID':'Movie ID'})

places_df = pd.read_csv('Places.csv', encoding = "ISO-8859-1", engine='python')
spells_df = pd.read_csv('Spells.csv', encoding = "ISO-8859-1", engine='python')



character_dialogue_df = characters_df.merge(dialogue_df, on='Character ID', how='left')
character_dialogue_places_df = character_dialogue_df.merge(places_df, on='Place ID', how='left')
character_dialogue_places_chapters_df = character_dialogue_places_df.merge(chapter_df, on='Chapter ID', how='left')
hp_df = character_dialogue_places_chapters_df.merge(movies_df, on='Movie ID', how='left')


spell_name = spells_df['Incantation']
hp_df["check"] = hp_df["Dialogue"].str.contains('|'.join(spell_name), na=False)

#nuanced change with string values to make string distance function work better by changing No and Fine strings
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('No.','place_holder_n.')
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('No!','place_holder_n!')
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('Fine.','place_holder_f.')
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('Fine!','place_holder_f!')

#distance string function
hp_df['Incantation'] = hp_df['Dialogue'].map(lambda x: difflib.get_close_matches(x, spells_df['Incantation'],cutoff=0.63, n=1))
hp_df['Incantation'] = hp_df['Incantation'].apply(lambda x: ''.join(map(str, x))) #0.67

#change No and Fine back
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('place_holder_n.','No.')
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('place_holder_n!','No!')
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('place_holder_f.','Fine.')
hp_df['Dialogue'] = hp_df['Dialogue'].str.replace('place_holder_f!','Fine!')

hp_df_final = hp_df.merge(spells_df, on='Incantation', how='left')

#need to download nltk for tokenizer
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download()

'''This is where application development begins using Dash framework'''

#define application
#app = dash.Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.SPACELAB])
#using for Heroku deployment
server = app.server

'''start UI development'''
image = 'url(https://i.postimg.cc/fbYVDWv8/hpwp7.jpg)'

app.layout = dbc.Container([
    html.Div(className='row',
             style={
                 'verticalAlign': 'middle',
                 'textAlign': 'center',
                 'background-image': image,
                 'background-size': 'cover',
                 'position': 'fixed',
                 'width': '100%',
                 'height': '100%',
                 'top': '0px',
                 'left': '0px',
                 'z-index': '-1'
             }
             ),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Harry Potter Character Analysis"),

            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0)',
                'text-align': 'center',
                'font-size': '250%',
                'color': 'white',
                'border': '0',
                'font-style': 'italic'
            }, outline=False)
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Dropdown(id='character-dd',
                                 options=[{'label': i, 'value': i} for i in hp_df_final['Character Name'].unique()],
                                 placeholder="Select a Character:")
                ]),
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.8)',
                'textAlign': 'center'
            })
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("Please Select a Character", className="card-text", id="message1-output")
                ], style={'color': 'white'})], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0)',
                'color': 'white',
                'text-align': 'center'})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Species:"),
                    html.P("Most Popular Places", className="card-text", id="species-output")
                ], style={'color': 'white',
                          'font-size': '100%'})
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.4)',
                'color': 'white',
                'text-align': 'center'
            })
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Gender:"),
                    html.P("Gender", className="card-text", id="gender-output")
                ], style={'color': 'white',
                          'font-size': '100%'})
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.4)',
                'color': 'white',
                'text-align': 'center'
            })
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("House:"),
                    html.P("House", className="card-text", id="house-output")
                ], style={'color': 'white',
                          'font-size': '100%'})
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.4)',
                'color': 'white',
                'text-align': 'center'
            })
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Most Used Spell:"),
                    html.P("Most Popular Places", className="card-text", id="spell-output")
                ], style={'color': 'white',
                          'font-size': '100%'})
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.4)',
                'color': 'white',
                'text-align': 'center'
            })
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Most Visited Place:"),
                    html.P("Most Popular Places", className="card-text", id="place-output")
                ], style={'color': 'white',
                          'font-size': '100%'})
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.4)',
                'color': 'white',
                'text-align': 'center'
            })
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Character Importance:"),
                    html.P("Character Importance", className="card-text", id="importance-output")
                ], style={'color': 'white',
                          'font-size': '100%'})
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.4)',
                'color': 'white',
                'text-align': 'center'
            })
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("How Many Lines?"),
                dbc.CardBody([
                    html.Div(id="lgraph-container", children=[
                        dcc.Graph(id='line-chart', figure={})
                    ]),
                ], style={
                    'backgroundColor': 'rgba(0,0,0,0)',
                    'text-align': 'center',
                    'color': 'white',
                    'font-size': '150%'
                })
            ], style={
                'backgroundColor': 'rgba(0,0,0,0.8)',
                'text-align': 'center',
                'color': 'white',
                'font-size': '150%'})
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Who's on your Mind?"),
                dbc.CardBody([
                    html.Div(id="graph-container", children=[
                        dcc.Graph(id='bar-chart', figure={})
                    ]),
                ], style={
                    'backgroundColor': 'rgba(0,0,0,0)',
                    'text-align': 'center',
                    'color': 'white',
                    'font-size': '150%'
                })
            ], style={
                'backgroundColor': 'rgba(0,0,0,0.8)',
                'text-align': 'center',
                'color': 'white',
                'font-size': '150%'})
        ])
    ])
], fluid=True)


'''Application Backend Development'''

@app.callback(
    Output(component_id='message1-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_message(input_value):
    if input_value is None:
        output = html.P(children=[html.Strong('Please select a character to see Character Facts!')],
                        style={'color': 'white',
                               'font-size': '150%'
                               })
    else:
        output = html.P(children=[html.Strong('Check out facts about ' + input_value + '!')],
                        style={'color': 'white',
                               'font-size': '150%'
                               })
    return output

'''
@app.callback(
    Output(component_id='message2-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_message2(input_value):
    if input_value is None:
        output = html.P(children=[html.Strong('Please select a character to see their Character Attributes!')],
                        style={'color': 'white'})
    else:
        output = html.P(children=[html.Strong("Check out " + input_value + "'s Attributes!")],
                        style={'color': 'white'})
    return output
'''

# Character Species
@app.callback(
    Output(component_id='species-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_species(input_value):
    if input_value is not None:
        output = hp_df_final[(hp_df_final['Character Name'] == input_value) & (~hp_df_final['Species'].isna())][
            'Species']
        if len(output) == 0:
            output = input_value + "'s Species hasn't been mentioned"
        else:
            output = " ".join(output.unique())
    else:
        output = ''
    return output


# Character Gender
@app.callback(
    Output(component_id='gender-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_gender(input_value):
    if input_value is not None:
        output = hp_df_final[(hp_df_final['Character Name'] == input_value) & (~hp_df_final['Gender'].isna())]['Gender']
        if len(output) == 0:
            output = input_value + "'s Gender hasn't been mentioned"
        else:
            output = " ".join(output.unique())
    else:
        output = ''
    return output


# Character House
@app.callback(
    Output(component_id='house-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_house(input_value):
    if input_value is not None:
        output = hp_df_final[(hp_df_final['Character Name'] == input_value) & (~hp_df_final['House'].isna())]['House']
        if len(output) == 0:
            output = input_value + "'s house hasn't been mentioned"
        else:
            output = " ".join(output.unique())
    else:
        output = ''
    return output


# Character Spells
@app.callback(
    Output(component_id='spell-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_spell(input_value):
    if input_value is not None:
        output_list = hp_df_final[(hp_df_final['Character Name'] == input_value) & (~hp_df_final['Spell Name'].isna())][
            'Incantation'].value_counts()
        if len(output_list) == 0:
            output_1 = input_value + " hasn't used a spell"
        else:
            output_1 = output_list.idxmax()
    else:
        output_1 = ''
    return output_1


# Character Places
@app.callback(
    Output(component_id='place-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_place(input_value):
    if input_value is not None:
        output_list = hp_df_final[(hp_df_final['Character Name'] == input_value) & (~hp_df_final['Place Name'].isna())][
            'Place Name'].value_counts()
        if len(output_list) == 0:
            output_1 = input_value + "'s locations haven't been noted"
        else:
            output_1 = output_list.idxmax()
    else:
        output_1 = ''
    return output_1


# Character Importance
@app.callback(
    Output(component_id='importance-output', component_property='children'),
    Input(component_id='character-dd', component_property='value')
)
def update_output_div_importance(input_value):
    char_imp = (hp_df_final.groupby(['Character Name'])['Chapter Name'].nunique() / hp_df_final[
        'Chapter Name'].nunique()) + ((hp_df_final['Character Name'].value_counts() / 7444))
    if input_value is not None:
        if char_imp[input_value] >= .10:
            output_1 = 'Very Important'
        elif (char_imp[input_value] < .10) & (char_imp[input_value] >= .03):
            output_1 = 'Important'
        elif (char_imp[input_value] < .03) & (char_imp[input_value] >= .01):
            output_1 = 'Somewhat Important'
        else:
            output_1 = 'Not as Important'
    else:
        output_1 = ''

    return output_1


# line chart
@app.callback(
    Output('lgraph-container', 'style'),
    [Input('character-dd', 'value')])
def hide_graph(input_value):
    if input_value:
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output(component_id='line-chart', component_property='figure'),
    Input(component_id='character-dd', component_property='value')
)
def line_chart_update(input_value):
    if input_value is not None:
        hp_df_final_agg = hp_df_final[hp_df_final['Character Name'] == input_value]
        grouped_df = hp_df_final_agg.groupby('Movie Title')
        grouped_df = grouped_df.agg({"Dialogue ID": "nunique"})
        grouped_df = grouped_df.reset_index()
        df_mapping = pd.DataFrame({
            'Movie Title': ["Harry Potter and the Philosopher's Stone",
                            "Harry Potter and the Chamber of Secrets",
                            "Harry Potter and the Prisoner of Azkaban",
                            "Harry Potter and the Goblet of Fire",
                            "Harry Potter and the Order of the Phoenix",
                            "Harry Potter and the Half-Blood Prince",
                            "Harry Potter and the Deathly Hallows Part 1",
                            "Harry Potter and the Deathly Hallows Part 2"]
        })
        sort_mapping = df_mapping.reset_index().set_index('Movie Title')
        grouped_df['idx'] = grouped_df['Movie Title'].map(sort_mapping['index'])
        grouped_df = grouped_df.set_index("idx")
        grouped_df = grouped_df.sort_index()

        # grouped_df = grouped_df.set_index(keys = movie_order, inplace= True)
        # grouped_df.sort_values(ascending=False, by= 'Dialogue ID')
        # grouped_df = grouped_df.sort_index()

        figure_line = px.line(grouped_df, x="Movie Title", y="Dialogue ID",
                              title='Line Graph', color_discrete_sequence=['lightblue'])
        figure_line.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
            yaxis_title="Count of Dialogue",
            font_color='white',
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            title={
                'text': "Amount of Dialogue spoken by " + input_value + " by Movie",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            hovermode="x")
        figure_line.update_traces(marker_color='lightblue', mode="markers+lines", hovertemplate=None)

    else:
        figure_line = go.Figure()
        figure_line.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            font_color="lightblue",
            annotations=[
                {
                    "text": "Please select a Character from the filter above to display!",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        )

    return figure_line


# bar chart
@app.callback(
    Output('graph-container', 'style'),
    [Input('character-dd', 'value')])
def hide_graph(input_value):
    if input_value:
        return {'display': 'block'}
    return {'display': 'none'}


# Bar Chart for Mentioned Names
@app.callback(
    Output(component_id='bar-chart', component_property='figure'),
    Input(component_id='character-dd', component_property='value')
)
def update_text_bar_chart(input_value):
    if input_value is not None:
        string_hp = hp_df_final['Character Name'].unique().tolist()
        hp_df_char = hp_df_final[hp_df_final['Character Name'] == input_value]['Dialogue'].tolist()
        sentence = " ".join(hp_df_char)
        new_tokens = word_tokenize(str(hp_df_char))
        new_tokens = [t.lower() for t in new_tokens]
        _new_stopwords_to_add = ['old', 'girl', 'boy', 'head', 'wizard', 'death']
        stopwords.words("english").extend(_new_stopwords_to_add)
        new_tokens = [t for t in new_tokens if t not in stopwords.words('english')]
        new_tokens = [t for t in new_tokens if t.isalpha()]
        lemmatizer = WordNetLemmatizer()
        new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
        counted = Counter(new_tokens)
        word_freq = pd.DataFrame(counted.items(), columns=['word', 'frequency']).sort_values(by='frequency',
                                                                                             ascending=False)

        hp_char_sep = [x.split() for x in string_hp]

        flat_list_1 = []
        for sublist in hp_char_sep:
            for item in sublist:
                flat_list_1.append(item)


        flat_list_hp = [[word.lower() for word in text.split()] for text in flat_list_1]

        flat_list_hp_final = []
        for sublist in flat_list_hp:
            for item in sublist:
                flat_list_hp_final.append(item)

        boolean_series = word_freq['word'].isin(flat_list_hp_final)

        word_freq['word'] = [each_string.capitalize() for each_string in word_freq['word']]
        filtered_freq = word_freq[boolean_series].head(7)

        figure = px.bar(
            data_frame=filtered_freq,
            x="word",
            y="frequency",
            # color="lightblue",
            opacity=0.9,
            labels={'word': 'Name', 'frequency': 'Frequency'})
        figure.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
            font_color='white',
            # marker_color='green',
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            title={
                'text': "Top Characters mentioned in Dialogue by " + input_value,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        figure.update_traces(marker_color='lightblue')


    else:
        figure = go.Figure()
        figure.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            font_color="lightblue",
            annotations=[
                {
                    "text": "Please select a Character from the filter above to display!",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        )

    return figure



'''Initiate Application'''
if __name__ == '__main__':
    app.run_server()
    #port = int(os.environ.get('PORT', 5000))
    #app_server.run(host='0.0.0.0', port=port)
