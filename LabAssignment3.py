import numpy as np

def dot_product(a,b): #A1
    if(len(a) !=len(b)):
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
    return norm

A=[2, 4, 6, 8]
B=[1, 3, 5, 7]
p,np_p=dot_product(A,B)
if isinstance(p, int):
    print(f"Dot product of {A} and {B} is {p}")
    print(f"Dot product of {A} and {B} using numpy is {np_p}")
else:
    print(p)
print("")
n=euclidean_norm(A)
if isinstance(n, int):
    print(f"Euclidean norm of {A} is {n}")
else:
    print(n)