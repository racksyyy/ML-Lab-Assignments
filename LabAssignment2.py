import pandas as pd
import numpy as np

def finding_rank_of_feature_matrix():
    df_purchase_data=pd.read_excel('Data/Lab Session Data.xlsx', sheet_name='Purchase data')
    clean=df_purchase_data.drop(columns=['Customer']).iloc[:, :4]
    vt=clean.to_numpy()
    rank=vt.linalg.matrix_rank()
    return rank

def cost_using_pseudoinverse():
    