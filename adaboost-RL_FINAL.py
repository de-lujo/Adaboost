import numpy as np
from sklearn import linear_model
from sklearn import metrics



def main():


    #menú con opciones para cargar dataset

    print ("Algoritmo Adaboost")
    print (" ")
    print ("1-Breast Cancer Wisconsin")
    print("Ingrese el dataset a clasificar: ")
    
    m= int(input())

    print("Ingrese la cantidad de iteraciones a ejecutar")

    iteraciones= int(input())


    #por mientras leer desde aquí
    if m==1: 

        file=open("wdbc.data")
        data_entrenar_x=[]
        data_entrenar_y=[]
        lista=[]
        data_testear_x=[]
        data_testear_y=[]
        
        i=1
        
        for line in file:
            lista=line.strip().split(",")
            largo=len(lista)
            
            #Obtenemos la clasificación de la clase
            
            if lista[1]=='M':

                    y=-1  #MALIGNO

            else:
                    y=1   #BENINGO

            #convertimos los vectores X e Y en una matriz.


            x=list(map(float,lista[2:31]))

            #tupla=tuple([array(x),y])


            #print (np.transpose(np.atleast_2d(x[:,1])))
            #print(x)
            
            if i<=426:
                
                data_entrenar_x.append(x)

                data_entrenar_y.append(y)

            else:

                data_testear_x.append(x)
                
                data_testear_y.append(y)

            i+=1 



        C=np.array(data_entrenar_x)
        D=np.array(data_entrenar_y)
        E=np.array(data_testear_x)
        F=np.array(data_testear_y)

        valor=adaboost(C,D,E,iteraciones)


        for i in range(len(E)):

            
            if valor[i]==1:

                    print ("",i+427,": Beningo\n")

            else :

                    print ("",i+427,": maligno\n")
    
    
    
    
    

def adaboost(X,Y,Z,M):                                      # M:números de iteraciones  X:arreglo de datos entrenar x  Y:arreglo de valores esperados de x
                                                            # Z:arreglo de datos testear x  

   W=(1.0/len(Y))*np.ones(Y.shape[0])                       #pesos iniciales

   reg=linear_model.LogisticRegression()                    #ocupamos regresión logística para nuestro clasificador débil

   bestClassifier=np.zeros((M,3))                           #Buscando el mejor clasificador

   
   for F in range(M):

        IndicatorList=[]                                    #lista de indicadores
        betaList=[]                                         #lista beta
        thresholdList=[]                                    #lista de umbrales
        bestVal=len(X)                                      #error
        bestFeat=len(X)                                     #mejor característica
        list_error=[]
        

        for i in range(X.shape[1]):                         #recorremos todos los datos de entrenamiento

    
                indata= np.transpose(np.atleast_2d(X[:,i])) #elegimos nuestra primera columna como caracteristica
                reg.fit(indata,Y)                           #asignamos a nuestra regresión lineal
                threshold=-1*reg.intercept_/reg.coef_       # calculamos nuestro umbral
                    
                Y_pred=reg.predict(indata)                  #predicimos nuestro posible valor
                Indicator = np.abs(Y_pred-Y)                #vemos nuestro rango del indicador de nuestro error
                errors=np.dot(Indicator,W)                  #calculamos nuestro peso inicial con el error

 
                if errors < bestVal:                        # el error es mayor que nuestro mínimo error
                        bestVal=errors                      # hacemos que el error sea nuestro minímo error
                        bestFeat=i                          # guardamos nuestro identificador donde estaba el error

                J=errors                                    #Calculamos nuestro beta

                list_error.append(J)
                beta=(1.0*J)/(1-J)
                IndicatorList.append(Indicator)             #Guardamos nuestro indicador, beta y umbral.
                betaList.append(beta)
                thresholdList.append(threshold)


        I=IndicatorList[bestFeat]                                    #calculamos la posición del indicador, el de beta y nuestro umbral
        beta= betaList[bestFeat]                            
        threshold=thresholdList[bestFeat]
        alpha=np.log(1.0/beta)                                      #calculamos nuestro alpha
        
        bestClassifier[F,:]=np.array([bestFeat,threshold,alpha])    #guardamos los datos obtenidos como el mejor clasificador débil
        
        for r in range(Y.shape[0]):   
              W[r]=W[r]*np.power(beta,1-I[r])        
                
        Wnorm=np.sum(W)                                                   #Sumamos nuestros pesos
        W=W/Wnorm                                                         #Normalizamos nuestros pesos


                                                                                                #Suma todos los umbrales alpha
        halfSumAlpha=0.5*np.sum(bestClassifier[:,2])

        x_datos=[]
        y_datos=[]
        for u in range(Z.shape[0]):
            
            sum=0
                
            for v in range(M):
                    
               if Z[u,bestClassifier[v,0]] > bestClassifier[v,1]:

                      sum=sum+bestClassifier[v,2]


                                                                                                # si es mayor es 1 , si es menor -1
               if sum > halfSumAlpha:
                                                                                                #  print ("Clasificado Maligno:  ",351+u)
                        y_datos.append(-1)
               else:
                                                                                                # print ("Clasificado Beningo:  ",351+u)
                        y_datos.append(1)

        

        
   guardar(X,list_error,"datosxy.txt")
        
   return y_datos
                        
def guardar(x_datos,y_datos,name_x):


        
        fo= open(name_x,"w")



        for i in range(len(y_datos)):

            
             fo.write(str(i+1))
             fo.write(" ")
             fo.write(str(y_datos[i]))
             fo.write("\n")

            
        fo.close()




    
