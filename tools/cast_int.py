import pandas as pd
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: mapping.csv, ...

df = pd.read_csv('mapping.csv')

df['population_locality'] = df['population_locality'].astype('Int64')
df['population_country'] = df['population_country'].astype('Int64')

df.to_csv('mapping.csv', index=False)
