import pandas as pd
import numpy as np
import os
import difflib


pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)


#movies_df = pd.read_csv('C:/Users/Willie/PycharmProjects/hpDashboard/Chapters.csv', encoding = "ISO-8859-1", engine='python')

os.chdir('C:/Users/Willie/PycharmProjects/hpApplication')
#os.chdir('C:/Users/Willie/PycharmProjects/HarryPotterApp')

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


print(hp_df_final.head(5))
