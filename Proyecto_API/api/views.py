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

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt


# Create your views here.
class CompanyView(View):
    ###Carga de datos###
    
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
    def getKmean(self):
        import os 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        data = pd.read_csv('C:/Users/HP/Documents/PythonPKmeans/Proyecto_API/api/Wholesale customers data.csv')
        data.shape
        data.isnull().sum()
        data.dtypes
        indices = [26, 176, 392]
        muestras = pd.DataFrame(data.loc[indices],
                                columns=data.keys()).reset_index(drop=True)
        data.drop(indices, axis=0)
        data = data.drop(['Region', 'Channel'], axis=1)
        muestras = muestras.drop(['Region', 'Channel'], axis=1)

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
        algoritmo = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10)
        algoritmo.fit(X)
        centroides, etiquetas= algoritmo.cluster_centers_,algoritmo.labels_
        muestra_prediccion = algoritmo.predict(muestras_escalada)
        for i, pred in enumerate(muestra_prediccion):
            print("Muestra",i,"se encuentra en el cluster",pred)
        ### GRAFICAR LOS DATOS JUNTO A LOS RESULTADOS ###
        # Se aplica la reducción de dimensionalidad a los datos
        from sklearn.decomposition import PCA
        modelo_pca = PCA(n_components = 3)
        modelo_pca.fit(X)
        pca = modelo_pca.transform(X) 

        #Se aplicar la reducción de dimsensionalidad a los centroides
        centroides_pca = modelo_pca.transform(centroides)
        print(muestra_prediccion)
        
        # Se define los colores de cada clúster
        colores = ['blue', 'red', 'green', 'orange', 'gray', 'brown']
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
        if(id>0):
            companies=list(Company.objects.filter(id=id).values())
            if len(companies)>0:
                company=companies[0]
                datos={"message":"Success","companies":company}
            else:
                datos={"message":"Company not found..."}
        else:
            companies= list(Company.objects.values())
            if len(companies)>0:
                datos={'message':"Success",'companies':companies}
            else:
                datos={'message':"Companies not found..."}
        return JsonResponse(datos)
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