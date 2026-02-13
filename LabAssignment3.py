import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



def dot_product(a,b): #A1
    if(len(a)!=len(b)):
        return "Enter vectors with same length"
    product=0
    for i in range(len(a)):
        product+=a[i]*b[i]
    a=np.array(a)
    b=np.array(b)
    np_product=np.dot(a,b)
    return product,np_product

def euclidean_norm(a): #A1
    if(len(a)==0):
        return "Enter a non-empty vector"
    norm=0
    for i in a:
        norm+=i**2
    norm=math.sqrt(norm)
    a=np.array(a)
    np_norm=np.linalg.norm(a)
    return norm,np_norm

def compute_mean(a):#A2
    return sum(a)/len(a)
 
def compute_variance(a):#A2
    mean_val=compute_mean(a)
    return sum((x-mean_val)**2 for x in a)/len(a)

def compute_std(a):#A2
    return compute_variance(a)**0.5

def mean_vector(m):#A2
    return np.array([compute_mean(m[:,i]) for i in range(m.shape[1])])

def std_vector(m):#A2
    return np.array([compute_std(m[:,i]) for i in range(m.shape[1])])

def compute_interclass_distance(a,b):
    return np.linalg.norm(a-b)

def plotting_histogram(ratings,bins):
    plt.hist(ratings,bins=bins)
    plt.xlabel("Movie Rating")
    plt.ylabel("Frequency")
    plt.title("Histogram of Movie Ratings")
    plt.show()

def minkowski_distance(a,b,p):  # A4
    distance=0
    for i in range(len(a)):
        distance+=abs(a[i]-b[i])**p
    return distance**(1/p)
def plot_minkowski(p_values,distances):
    plt.plot(p_values,distances,marker='o')
    plt.xlabel("Value of p")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p")
    plt.show()

def euclidean_distance(a,b):  # A10
    dist=0
    for i in range(len(a)):
        dist+=(a[i]-b[i])**2
    return math.sqrt(dist)

def knn_predict_single(X_train,y_train,test_vector,k):  # A10
    distances=[]
    for i in range(len(X_train)):
        d=euclidean_distance(X_train[i],test_vector)
        distances.append((d,y_train[i]))
    distances.sort(key=lambda x:x[0])
    k_nearest=distances[:k]
    votes=[label for (_,label) in k_nearest]
    return max(set(votes),key=votes.count)

def knn_predict(X_train,y_train,X_test,k):  # A10
    predictions=[]
    for i in range(len(X_test)):
        pred=knn_predict_single(X_train,y_train,X_test[i],k)
        predictions.append(pred)
    return np.array(predictions)

def k_vs_accuracy_plot(k_values, test_accuracies, train_accuracies=None):
    plt.plot(k_values, test_accuracies, marker='o', label='Test Accuracy', color='blue')
    plt.plot(k_values, train_accuracies, marker='o', label='Train Accuracy', color='red')
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("k vs Accuracy for k-NN Classifier")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_confusion_elements(y_true, y_pred):  # A13
    TP=FP=TN=FN=0

    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:
            TP+=1
        elif y_true[i]==0 and y_pred[i]==1:
            FP+=1
        elif y_true[i]==0 and y_pred[i]==0:
            TN+=1
        elif y_true[i]==1 and y_pred[i]==0:
            FN+=1

    return TP, FP, TN, FN

def accuracy_score_custom(y_true, y_pred):  # A13
    TP,FP,TN,FN=compute_confusion_elements(y_true, y_pred)
    return (TP+TN)/(TP+TN+FP+FN)

def precision_score_custom(y_true, y_pred):  # A13
    TP,FP,_,_=compute_confusion_elements(y_true, y_pred)
    return TP/(TP+FP) if (TP+FP)!=0 else 0

def recall_score_custom(y_true, y_pred):  # A13
    TP,_,_,FN=compute_confusion_elements(y_true, y_pred)
    return TP/(TP+FN) if (TP+FN)!=0 else 0

def fbeta_score_custom(y_true, y_pred, beta):  # A13
    precision=precision_score_custom(y_true, y_pred)
    recall=recall_score_custom(y_true, y_pred)

    if precision==0 and recall==0:
        return 0

    return (1+beta**2)*(precision*recall)/((beta**2*precision)+recall)

def train_matrix_inversion_classifier(X, y):  # A14
    X_bias=np.hstack((np.ones((X.shape[0],1)),X))
    y_transformed=np.where(y==0,-1,1)

    w=np.linalg.pinv(X_bias.T@X_bias)@X_bias.T@y_transformed
    return w

def predict_matrix_inversion_classifier(X, w):  # A14
    X_bias=np.hstack((np.ones((X.shape[0],1)),X))
    y_pred=X_bias@w
    return np.where(y_pred>=0,1,0)


if __name__=="__main__":
    #A1
    print("Question A1: ")
    A=[2,4,6,8]
    B=[1,3,5,7]
    p,np_p=dot_product(A,B)
    if isinstance(p,int):
        print(f"Dot product of {A} and {B} is {p}")
        print(f"Dot product of {A} and {B} using numpy is {np_p}")
    else:
        print(p)
    print("")

    n,np_norm=euclidean_norm(A)
    if isinstance(n,float):
        print(f"Euclidean norm of {A} is {n}")
        print(f"Euclidean norm of {A} using numpy is {np_norm}")
    else:
        print(n)

    #A2
    print("")
    print("\nQuestion A2: ")
    data=pd.read_csv("Data/IMDB-Movie-Data.csv")
    features=data[["Rating","Votes","Runtime (Minutes)","Revenue (Millions)","Metascore"]]
    features=features.dropna()
    labels=np.where(features["Rating"]<6.5,0,1)
    X=features.values
    class_0=X[labels==0]
    class_1=X[labels==1]

    centroid_0=mean_vector(class_0)
    centroid_1=mean_vector(class_1)
    spread_0=std_vector(class_0)
    spread_1=std_vector(class_1)
    interclass_distance=compute_interclass_distance(centroid_0,centroid_1)

    print("Centroid of Class 0:", centroid_0)
    print("Centroid of Class 1:", centroid_1)

    print("Spread (Std Dev) of Class 0:", spread_0)
    print("Spread (Std Dev) of Class 1:", spread_1)

    print("Interclass Distance Between Centroids:", interclass_distance)

    #A3
    print("")
    print("\nQuestion A3: ")
    ratings=data["Rating"].dropna().values
    mean_rating=np.mean(ratings)
    variance_rating=np.var(ratings)
    bins=10
    hist_values,bin_edges=np.histogram(ratings,bins=bins)
    plotting_histogram(ratings,bins)
    print("Mean of Rating:",mean_rating)
    print("Variance of Rating:",variance_rating)
    
    
    # A4
    print("")
    print("\nQuestion A4: ")
    vec1=X[0]
    vec2=X[1]
    p_values=list(range(1,11))
    distances=[]
    for i in p_values:
        d=minkowski_distance(vec1,vec2,i)
        distances.append(d)
        print(f"Minkowski Distance for p={i}: {d}")

    plot_minkowski(p_values, distances)

    # A5
    print("")
    print("\nQuestion A5: ")
    p=3
    own_distance=minkowski_distance(vec1,vec2,p)
    scipy_distance=minkowski(vec1,vec2,p)
    print(f"Own Minkowski Distance (p={p}):",own_distance)
    print(f"SciPy Minkowski Distance (p={p}):",scipy_distance)
 
    # A6
    print("")
    print("\nQuestion A6: ")

    X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=0.3,random_state=42)

    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

    # A7
    print("")
    print("\nQuestion A7: ")
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    print("kNN classifier trained with k = 3")

    # A8
    print("")
    print("\nQuestion A8: ")
    test_accuracy = neigh.score(X_test, y_test)

    print("Test Accuracy of k-NN classifier:", test_accuracy)

    # A9
    print("")
    print("\nQuestion A9: ")
    y_pred=neigh.predict(X_test)
    print("Predicted labels for test set (first 10):",y_pred[:10])
    print("Actual labels for test set (first 10):",y_test[:10])
    test_vector=X_test[0].reshape(1,-1)
    single_prediction=neigh.predict(test_vector)

    print("Prediction for a single test vector:", single_prediction[0])
 
    # A10
    print("")
    print("\nQuestion A10: ")
    k=3
    y_pred_custom=knn_predict(X_train,y_train,X_test,k)
    custom_accuracy=np.mean(y_pred_custom==y_test)

    print("Accuracy of custom kNN classifier:", custom_accuracy)
    print("Accuracy of sklearn kNN classifier:", test_accuracy)
 
    # A11
    print("")
    print("\nQuestion A11: ")
    k_values=list(range(1,101))
    test_accuracies=[]
    train_accuracies=[]
    for k in k_values:
        y_pred_test=knn_predict(X_train,y_train,X_test,k)
        y_pred_train=knn_predict(X_train,y_train,X_train,k)
        test_acc=np.mean(y_pred_test==y_test)
        train_acc=np.mean(y_pred_train==y_train)
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)
        print(f"Accuracy for k = {k}: Test={test_acc:.4f}, Train={train_acc:.4f}")

    k_vs_accuracy_plot(k_values, test_accuracies, train_accuracies)

    # A12
    print("")
    print("\nQuestion A12: ")
    y_train_pred=neigh.predict(X_train)
    y_test_pred=neigh.predict(X_test)

    cm_train=confusion_matrix(y_train,y_train_pred)
    cm_test=confusion_matrix(y_test,y_test_pred)

    print("Confusion Matrix (Training Data):\n", cm_train)
    print("Confusion Matrix (Test Data):\n", cm_test)

    train_accuracy=np.mean(y_train_pred==y_train)
    train_precision=precision_score(y_train,y_train_pred)
    train_recall=recall_score(y_train,y_train_pred)
    train_f1=f1_score(y_train,y_train_pred)

    test_precision=precision_score(y_test,y_test_pred)
    test_recall=recall_score(y_test,y_test_pred)
    test_f1=f1_score(y_test,y_test_pred)

    print("\nTraining Metrics:")
    print("Accuracy:", train_accuracy)
    print("Precision:", train_precision)
    print("Recall:", train_recall)
    print("F1-Score:", train_f1)
    print("\nTest Metrics:")
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1-Score:", test_f1)

    # A13
    print("")
    print("\nQuestion A13: ")
    beta=1

    acc_custom=accuracy_score_custom(y_test,y_test_pred)
    prec_custom=precision_score_custom(y_test,y_test_pred)
    rec_custom=recall_score_custom(y_test,y_test_pred)
    fbeta_custom=fbeta_score_custom(y_test,y_test_pred,beta)

    print("Custom Accuracy:", acc_custom)
    print("Custom Precision:", prec_custom)
    print("Custom Recall:", rec_custom)
    print(f"Custom F{beta}-Score:", fbeta_custom)

        # A14
    print("")
    print("\nQuestion A14: ")

    w=train_matrix_inversion_classifier(X_train,y_train)
    y_test_pred_matrix=predict_matrix_inversion_classifier(X_test,w)
    matrix_accuracy=np.mean(y_test_pred_matrix==y_test)

    print("Accuracy of Matrix Inversion Classifier:", matrix_accuracy)
    print("Accuracy of kNN Classifier:", test_accuracy)


