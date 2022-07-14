from django.http import JsonResponse
from django.views import View
from .models import Company
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import json
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
# Create your views here.
class DataView(View):
    ###Carga de datos###
    
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
    @csrf_exempt
    def getTdData(request):
        jd=json.loads(request.body)
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        dataframe = pd.read_csv("C:/Users/HP/Documents/PythonPKmeans/Proyecto_API/api/Horoscopo-csv.csv")
        #dataframe.head()
        #dataframe.describe()
        print(jd)
        X = np.array(dataframe[[jd['columna1'],jd['columna2'],jd['columna3']]])
        y = np.array(dataframe['SignoZodiacal'])
        Nc = range(1, 20)
        kmeans = [KMeans(n_clusters=i) for i in Nc]
        score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
        kmeans = KMeans(n_clusters=5).fit(X)
        centroids = kmeans.cluster_centers_
        labels = kmeans.predict(X)
        signos=['Aries','Tauro','Geminis','Cancer','Leo','Virgo','Libra','Escorpio','Sagitario','Capricornio','Acuario','Piscis']
        colores=['red','green','blue','cyan','yellow']
        asignar=[]
        json_dump=[]
        rojo=[]
        verde=[]
        azul=[]
        cyan=[]
        amarillo=[]
        count=0
        xMax=X.max(axis=0)
        for row in labels:
            #print(X[count][0])
            if row ==0:
                rojo.append({"x": int(X[count][0]) ,"y":int(X[count][1]),"z":int(X[count][2]),"name":signos[y[count]-1],"color":colores[row]})
                asignar.append(colores[row])
            elif row==1:
                verde.append({"x": int(X[count][0]) ,"y":int(X[count][1]),"z":int(X[count][2]),"name":signos[y[count]-1],"color":colores[row]})
                asignar.append(colores[row])
            elif row==2:
                azul.append({"x": int(X[count][0]) ,"y":int(X[count][1]),"z":int(X[count][2]),"name":signos[y[count]-1],"color":colores[row]})
                asignar.append(colores[row])              
            elif row==3:
                cyan.append({"x": int(X[count][0]) ,"y":int(X[count][1]),"z":int(X[count][2]),"name":signos[y[count]-1],"color":colores[row]})
                asignar.append(colores[row])
            elif row==4:
                amarillo.append({"x": int(X[count][0]) ,"y":int(X[count][1]),"z":int(X[count][2]),"name":signos[y[count]-1],"color":colores[row]})
                asignar.append(colores[row])
            count+=1    
        # Getting the cluster centers
        C = kmeans.cluster_centers_      
        json_dump = json.dumps({"rojo": rojo,"verde": verde,"azul": azul,"cyan": cyan,"amarillo": amarillo,"centroides":C.tolist(),"colores":colores
        ,"xMax":int(xMax[0]),"yMax":int(xMax[1]),"zMax":int(xMax[2])})
        print(json_dump)
        return JsonResponse(json_dump,safe=False)
    def getKmean(self):
        import os 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        data = pd.read_csv('C:/Users/HP/Documents/PythonPKmeans/Proyecto_API/api/Horoscopo-csv.csv')
        data.shape
        data.isnull().sum()
        data.dtypes
        indices = [2,10,20]
        muestras = pd.DataFrame(data.loc[indices],
                                columns=data.keys()).reset_index(drop=True)
        data.drop(indices, axis=0)
        data = data.drop(['Marca temporal', '¿Cuál es tu signo Zodiacal?'], axis=1)
        muestras = muestras.drop(['Marca temporal', '¿Cuál es tu signo Zodiacal?'], axis=1)
        data_escalada = preprocessing.Normalizer().fit_transform(data)
        muestras_escalada=preprocessing.Normalizer().fit_transform(muestras)
        X = data_escalada.copy()
        inercia = []
        for i in range(1, 20):
            algoritmo = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
            algoritmo.fit(X)
            inercia.append(algoritmo.inertia_)
        plt.figure(figsize=(10, 6))
        plt.title('Metodo del Codo')
        plt.xlabel('No. de clusters')
        plt.ylabel('Inercia')
        plt.plot(list(range(1, 20)), inercia, marker='o')
        plt.show()
        algoritmo = KMeans(n_clusters=12, init='k-means++', max_iter=300, n_init=10)
        algoritmo.fit(X)
        print(algoritmo.cluster_centers_)
        print(algoritmo.labels_)
        centroides, etiquetas= algoritmo.cluster_centers_,algoritmo.labels_
        muestra_prediccion = algoritmo.predict(muestras_escalada)
        print(muestra_prediccion)
        for i, pred in enumerate(muestra_prediccion):
            print("Muestra",i,"se encuentra en el cluster",pred)
        ### GRAFICAR LOS DATOS JUNTO A LOS RESULTADOS ###
        # Se aplica la reducción de dimensionalidad a los datos
        from sklearn.decomposition import PCA
        modelo_pca = PCA(n_components = 2)
        modelo_pca.fit(X)
        pca = modelo_pca.transform(X)
        #Se aplicar la reducción de dimsensionalidad a los centroides
        centroides_pca = modelo_pca.transform(centroides)
        # Se define los colores de cada clúster
        colores = ['blue', 'red', 'green', 'orange', 'gray', 'brown','yellow','pink','blue', 'darkblue', 'darkgreen','darkred']
        #Se asignan los colores a cada clústeres
        colores_cluster = [colores[etiquetas[i]] for i in range(len(pca))]

        #Se grafica los componentes PCA
        plt.scatter(pca[:, 0], pca[:, 1], c = colores_cluster, 
                    marker = 'o',alpha = 0.4)

        #Se grafican los centroides
        plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
                    marker = 'x', s = 100, linewidths = 3, c = colores)

        #Se guadan los datos en una variable para que sea fácil escribir el código
        xvector = modelo_pca.components_[0] * max(pca[:,0])
        yvector = modelo_pca.components_[1] * max(pca[:,1])
        columnas = data.columns

        #Se grafican los nombres de los clústeres con la distancia del vector
        for i in range(len(columnas)):
            #Se grafican los vectores
            plt.arrow(0, 0, xvector[i], yvector[i], color = 'black', 
                    width = 0.0005, head_width = 0.02, alpha = 0.75)
            #Se colocan los nombres
            plt.text(xvector[i], yvector[i], list(columnas)[i], color='black', 
                    alpha=0.75)

        plt.show()
        return JsonResponse({"mensaje":"Success"})
    def get(self,request,id=0):
        data=[
                [1, 6, 5], [8, 7, 9], [1, 3, 4], [4, 6, 8], [5, 7, 7], [6, 9, 6],
                [7, 0, 5], [2, 3, 3], [3, 9, 8], [3, 6, 5], [4, 9, 4], [2, 3, 3],
                [6, 9, 9], [0, 7, 0], [7, 7, 9], [7, 2, 9], [0, 6, 2], [4, 6, 7],
                [3, 7, 7], [0, 1, 7], [2, 8, 6], [2, 3, 7], [6, 4, 8], [3, 5, 9],
                [7, 9, 5], [3, 1, 7], [4, 4, 2], [3, 6, 2], [3, 1, 6], [6, 8, 5],
                [6, 6, 7], [4, 1, 1], [7, 2, 7], [7, 7, 0], [8, 8, 9], [9, 4, 1],
                [8, 3, 4], [9, 8, 9], [3, 5, 3], [0, 2, 4], [6, 0, 2], [2, 1, 3],
                [5, 8, 9], [2, 1, 1], [9, 7, 6], [3, 0, 2], [9, 9, 0], [3, 4, 8],
                [2, 6, 1], [8, 9, 2], [7, 6, 5], [6, 3, 1], [9, 3, 1], [8, 9, 3],
                [9, 1, 0], [3, 8, 7], [8, 0, 0], [4, 9, 7], [8, 6, 2], [4, 3, 0],
                [2, 3, 5], [9, 1, 4], [1, 1, 4], [6, 0, 2], [6, 1, 6], [3, 8, 8],
                [8, 8, 7], [5, 5, 0], [3, 9, 6], [5, 4, 3], [6, 8, 3], [0, 1, 5],
                [6, 7, 3], [8, 3, 2], [3, 8, 3], [2, 1, 6], [4, 6, 7], [8, 9, 9],
                [5, 4, 2], [6, 1, 3], [6, 9, 5], [4, 8, 2], [9, 7, 4], [5, 4, 2],
                [9, 6, 1], [2, 7, 3], [4, 5, 4], [6, 8, 1], [3, 4, 0], [2, 2, 6],
                [5, 1, 2], [9, 9, 7], [6, 9, 9], [8, 4, 3], [ 4, 1, 7], [6, 2, 5],
                [0, 4, 9], [3, 5, 9], [6, 9, 1], [1, 9, 2]]
        # if(id>0):
        #     companies=list(Company.objects.filter(id=id).values())
        #     if len(companies)>0:
        #         company=companies[0]
        #         datos={"message":"Success","companies":company}
        #     else:
        #         datos={"message":"Company not found..."}
        # else:
        #     companies= list(Company.objects.values())
        #     if len(companies)>0:
        #         datos={'message':"Success",'companies':companies}
        #     else:
        #         datos={'message':"Companies not found..."}
        return JsonResponse({"data":data})
    def post(self,request):
        #print(request.body)
        jd=json.loads(request.body)
        #print(jd)
        Company.objects.create(name=jd['name'],website=jd['website'],foundation=jd['foundation'])
        datos={'message':"Success"}
        return JsonResponse(datos)
    def put(self,request,id):
        jd=json.loads(request.body)
        companies=list(Company.objects.filter(id=id).values())
        if len(companies)>0:
            company=Company.objects.get(id=id)
            company.name=jd['name']
            company.website=jd['website']
            company.foundation=jd['foundation']
            company.save()
            datos={"message":"Success"}
        else:
            datos={'message':"Companies not found..."}
        
        return JsonResponse(datos)
    def delete(self,request,id):
        companies=list(Company.objects.filter(id=id).values())
        if len(companies)>0:
            Company.objects.filter(id=id).delete()
            datos={"message":"Success"}
        else:
            datos={'message':"Companies not found..."}
        return JsonResponse(datos)