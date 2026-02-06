from sklearn.metrics import classification_report
import  pandas as pd
def confusion_matrix():
    df_training=pd.read_excel("Data/KNN_Confusion_Matrix.xlsx", sheet_name= "Training Data")
    df_test=pd.read_excel("Data/KNN_Confusion_Matrix.xlsx", sheet_name="Test Data")
    return df_training, df_test

def Precision():
    train, test=confusion_matrix()
    TP_train=train.loc(1, "Positive")
    TP_test=test.loc(1,"Positive")
    FP_train=train.loc(1, "Negative")
    FP_test=test.loc(1,"Negative")
    TN_train=train.loc(2, "Positive")
    TN_test=test.loc(2, "Positive")
    FN_train=train.loc(2, "Negative")
    FN_test=test.loc(2,"Negative")
    precision_test=TP_test/(TP_test+FN_test)
    precision_train=TP_train/(TP_train+FN_train)
    return precision_train, precision_test

#def Recall():

#def F1_Score():


if __name__ == "__main__":
    #A1
    print("Question A1")
    train, test=confusion_matrix()
    print("Confusion Matrix for Trainning Data:")
    print(train)
    print("")
    print("Confusion Matrix for Test Data:")
    print(test)
    print("")
    precision_train,precision_test=Precision()
    print("Precision of Training Data: ", precision_train)
    print("Precision of Test Data: ", precision_test)

