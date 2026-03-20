import pandas as df
df=pd.read_excel("Data/IMDB-Movie-Data.xlsx")

def entropy_clalculate():
    df_entropy=pd.cut(df['Rating'],bins=[0,2,4,6,8,10],labels=['0-2','2-4','4-6','6-8','8-10'])
    print(df_entropy)