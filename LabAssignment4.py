from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import  pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def confusion_matrix():
    df=pd.read_excel("Data/KNN_Confusion_Matrix.xlsx", sheet_name="Values", index_col=0)
    cm_test=pd.read_excel("Data/KNN_Confusion_Matrix.xlsx",sheet_name="CM_Test")
    cm_train=pd.read_excel("Data/KNN_Confusion_Matrix.xlsx",sheet_name="CM_Train")
    return df, cm_test, cm_train

def Precision_Recall_F1Score():
    df,x,y=confusion_matrix()
    TP_train=df.at["Train","TP"]
    TP_test=df.at["Test","TP"]
    FP_train=df.at["Train","FP"]   
    FP_test=df.at["Test","FP"]
    TN_train=df.at["Train","TN"]
    TN_test=df.at["Test","TN"]
    FN_train=df.at["Train","FN"]
    FN_test=df.at["Test","FN"]
    precision_test=TP_test/(TP_test+FP_test)
    precision_train=TP_train/(TP_train+FP_train)
    recall_test=TP_test/(TP_test+FP_test)
    recall_train=TP_train/(TP_train+FP_train)
    F1_Score_test = 2*(precision_test*recall_test)/(precision_test+recall_test)
    F1_Score_train = 2*(precision_train*recall_train)/(precision_train+recall_train)
    return precision_train, precision_test, recall_train, recall_test, F1_Score_test, F1_Score_train

def questionA3(check):
    X,Y = np.random.uniform(0,10,(2,20))
    if (check):
        return X,Y
    colors = np.where(X < Y, 'blue', 'red')
    plt.scatter(X,Y, c=colors)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Training Data Scatter Plot")
    plt.show()

def question4():
    x_train,y_train=questionA3(1)
    labels_train = np.where(x_train<y_train, 0, 1)
    train_points = np.column_stack((x_train,y_train))
    x_test = np.arange(0, 10.1, 0.1)
    y_test = np.arange(0, 10.1, 0.1)
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_points, labels_train)
    xx, yy = np.meshgrid(x_test, y_test)
    test_points = np.column_stack((xx.ravel(), yy.ravel()))
    predictions = knn.predict(test_points)
    colors = np.where(predictions == 0, 'blue', 'red')
    plt.figure()
    plt.scatter(test_points[:, 0], test_points[:, 1], c=colors, s=5)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("kNN (k=3) Classification on Test Data")
    plt.show()

def question5():
    x_train,y_train=questionA3(1)
    labels_train = np.where(x_train<y_train, 0, 1)
    train_points = np.column_stack((x_train,y_train))
    x_test = np.arange(0, 10.1, 0.1)
    y_test = np.arange(0, 10.1, 0.1)
    for k in range(1,11):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_points, labels_train)
        xx, yy = np.meshgrid(x_test, y_test)
        test_points = np.column_stack((xx.ravel(), yy.ravel()))
        predictions = knn.predict(test_points)
        colors = np.where(predictions == 0, 'blue', 'red')
        plt.figure()
        plt.scatter(test_points[:, 0], test_points[:, 1], c=colors, s=5)
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.title(f"kNN (k={k}) Classification on Test Data")
        plt.show()

def GridSearchCVSearch():
    x_train,y_train=questionA3(1)
    labels_train = np.where(x_train<y_train, 0, 1)
    train_points = np.column_stack((x_train,y_train))
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 11))}
    grid = GridSearchCV(knn, param_grid, cv=5)
    grid.fit(train_points, labels_train)
    print("Best k value:", grid.best_params_['n_neighbors'])
    print("Best cross-validation accuracy:", grid.best_score_)



if __name__ == "__main__":
    #A1
    print("Question A1")
    x,test, train= confusion_matrix()
    print("Confusion Matrix for Trainning Data:")
    print(train)
    print("")
    print("Confusion Matrix for Test Data:")
    print(test)
    print("")
    precision_train, precision_test, recall_train, recall_test, F1_Score_test, F1_Score_train=Precision_Recall_F1Score()
    print("Precision of Training Data: ", precision_train)
    print("Precision of Test Data: ", precision_test)
    print("Recall of Training Data: ", recall_train)
    print("Recall of Test Data: ", recall_test)
    print("F1-Score of Training Data: ", F1_Score_train)
    print("F1-Score of Test Data: ", F1_Score_test)

    #A2
    print("Question A2")
    print("")

    #A3
    print("Question A3")
    questionA3(0)
    print("Plotted")
    print("")

    #A4
    print("Question 4")
    question4()
    print()

    #A5
    print("Question 5")
    question5()
    print()

    #A6
    print("Question 6")
    print("Project data is was used for the above questions")
    print()

    #A7
    print("Question 7")
    GridSearchCVSearch()




