from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import  pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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

def question6():
    #A3
    df = pd.read_csv("Data/IMDB-Movie-Data.csv")
    df = df[['Rating', 'Revenue (Millions)']].dropna()
    df['Class'] = np.where(df['Rating'] >= 7, 1, 0)
    X_train = df[['Rating', 'Revenue (Millions)']].values
    y_train = df['Class'].values
    plt.figure()
    plt.scatter(X_train[y_train == 0][:, 0],X_train[y_train == 0][:, 1])
    plt.scatter(X_train[y_train == 1][:, 0],X_train[y_train == 1][:, 1])
    plt.xlabel("Rating")
    plt.ylabel("Revenue (Millions)")
    plt.title("Training Data (Blue = Class 0, Red = Class 1)")
    plt.show()

    #A4
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),np.arange(y_min, y_max, 0.2))
    X_test = np.c_[xx.ravel(), yy.ravel()]
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    plt.figure()
    plt.scatter(X_test[y_pred == 0][:, 0],X_test[y_pred == 0][:, 1], s=5)
    plt.scatter(X_test[y_pred == 1][:, 0],X_test[y_pred == 1][:, 1], s=5)
    plt.xlabel("Rating")
    plt.ylabel("Revenue (Millions)")
    plt.title("kNN Classification (k = 3)")
    plt.show()

    #A5
    for k in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred_k = knn.predict(X_test_scaled)
        plt.figure()
        plt.scatter(X_test[y_pred_k == 0][:, 0], X_test[y_pred_k == 0][:, 1], s=5)
        plt.scatter(X_test[y_pred_k == 1][:, 0],X_test[y_pred_k == 1][:, 1], s=5)
        plt.xlabel("Rating")
        plt.ylabel("Revenue (Millions)")
        plt.title(f"kNN Classification (k = {k})")
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
    print("Question A4")
    question4()
    print()

    #A5
    print("Question A5")
    question5()
    print()

    #A6
    print("Question A6")
    question6()
    print()

    #A7
    print("Question A7")
    GridSearchCVSearch()




