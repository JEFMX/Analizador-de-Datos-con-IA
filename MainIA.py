# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np #para crear vectores y matrices de n dimensiones
import pandas as pd
import math
import matplotlib.pyplot as plt #Para gráficar
import seaborn as sb   #Biblioteca para visualización de datos basado en matplotlib
import scipy.cluster.hierarchy as shc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from apyori import apriori
from PyQt5.QtWidgets import *
from Analizador import *
from scipy.spatial import distance
from PyQt5 import uic

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

#Var Global
Data = 0
x = 0
y = 0
Transacciones = []
Confianza = 0
Soporte = 0
Lift = 0
Tamanio = 0
AuxCambio = 0
Texture = 0
Area = 0
Compactness = 0
Concavity = 0
FractalDim = 0
Symmetry = 0
DataCluster = 0

class AnalizadorDataApp(Ui_QMainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #Eventos que se presentan
        self.pB_Cargar.clicked.connect(self.leerConCajas)
        self.pB_Confirmar.clicked.connect(self.Algoritmos)
        self.pB_Enviar.clicked.connect(self.LeerParametrosApriori)
        self.pB_Enviar_2.clicked.connect(self.LeerParametrosCorrelacional)
        self.pB_Nuevo.clicked.connect(self.NuevoAnalisis)
        self.pB_Cambiar.clicked.connect(self.Cambiar)
        self.pB_Enviar_5.clicked.connect(self.LeerParametrosDiagnostico)
        self.pB_Nuevo_2.clicked.connect(self.NuevoDiagnostico)
        self.pB_Euclidiana.clicked.connect(self.Euclidiana)
        self.pB_Manhattan.clicked.connect(self.Manhattan)
        self.pB_Minkowsky.clicked.connect(self.Minkowsky)
        self.pB_Chebyshev.clicked.connect(self.Chebyshev)
        self.pB_Enviar_4.clicked.connect(self.ClusteringParticional)

        #CONFIG INCIAL ETIQUETAS, BOTONES, CAJASDETEXTO , ETC.
        self.label_2.setEnabled(False)
        self.label_3.setEnabled(False)
        self.label_4.setEnabled(False)
        self.label_5.setEnabled(False)
        self.label_6.setEnabled(False)
        self.label_7.setEnabled(False)
        self.label_8.setEnabled(False)
        self.label_9.setEnabled(False)
        self.label_10.setEnabled(False)
        self.label_11.setEnabled(False)
        self.label_12.setHidden(True)
        self.label_13.setHidden(True)
        self.label_14.setEnabled(False)
        self.label_15.setEnabled(False)
        self.label_16.setEnabled(False)
        self.label_17.setEnabled(False)
        self.label_18.setEnabled(False)
        self.label_19.setHidden(True)
        self.label_20.setEnabled(False)
        self.label_22.setEnabled(False)
        self.label_23.setEnabled(False)
        self.lbl_Algoritmo.setEnabled(False)
        self.pB_Enviar.setEnabled(False)
        self.pB_Enviar_2.setEnabled(False)
        self.pB_Euclidiana.setEnabled(False)
        self.pB_Chebyshev.setEnabled(False)
        self.pB_Manhattan.setEnabled(False)
        self.pB_Minkowsky.setEnabled(False)
        self.pB_Enviar_4.setEnabled(False)
        self.pB_Confirmar.setEnabled(False)
        self.cb_Algoritmo.setEnabled(False)
        self.LE_Elemento_1.setEnabled(False)
        self.LE_Elemento_2.setEnabled(False)
        self.LE_Confianza.setEnabled(False)
        self.LE_Soporte.setEnabled(False)
        self.LE_Tamanio.setEnabled(False)
        self.LE_lift.setEnabled(False)
        self.lbl_Titulo_2.setEnabled(False)
        self.LE_Texture.setEnabled(False)
        self.LE_Compactness.setEnabled(False)
        self.LE_Symmetry.setEnabled(False)
        self.LE_FractalDim.setEnabled(False)
        self.LE_Concavity.setEnabled(False)
        self.pB_Enviar_5.setEnabled(False)
        self.lb_Diagnostico.setEnabled(False)
        self.pB_Nuevo.setEnabled(False)
        self.pB_Nuevo_2.setEnabled(False)
        self.pB_ok.setEnabled(False)
        self.cb_Algoritmo_2.setEnabled(False)
        self.show()

    def leerConCajas(self): #Funcion para la lectura de un archivos tipo .txt, .csv o .xls
        if not self.LE_NombreArchivo.text():
            #Mensaje de error en caso de no introducir ningun nombre de archivo
            QMessageBox.critical(self, "ERROR", "No se introdujo nombre de archivo")
        else:
            try:
                #Obtencion del nombre del archivo
                NombreArchivo = self.LE_NombreArchivo.text()
                #Obtencion de la extensión del archivo
                Extension = self.LE_Extension.text()
                Extension.lower
                global Data
                if Extension == ".txt" :
                    File =  NombreArchivo + Extension
                    Data = pd.read_table(File)
                    DataStr = str(Data)
                    self.txt_Texto.setText(DataStr)
                    self.label_19.setHidden(False) #Se activan iconos de que se leyo correctamente el archivo
                    self.cb_Algoritmo.setEnabled(True)
                    self.lbl_Algoritmo.setEnabled(True)
                    self.pB_Confirmar.setEnabled(True)
                    self.lbl_SolicitoArchivo.setEnabled(False)
                    self.LE_NombreArchivo.setEnabled(False)
                    self.label.setEnabled(False)
                    self.LE_Extension.setEnabled(False)
                    self.pB_Cargar.setEnabled(False)
                if Extension == ".csv":
                    File = NombreArchivo + Extension
                    Data = pd.read_csv(File, header = None)
                    DataStr = str(Data)
                    self.txt_Texto.setText(DataStr)
                    self.label_19.setHidden(False) #Se activan iconos de que se leyo correctamente el archivo
                    self.cb_Algoritmo.setEnabled(True)
                    self.lbl_Algoritmo.setEnabled(True)
                    self.pB_Confirmar.setEnabled(True)
                    self.lbl_SolicitoArchivo.setEnabled(False)
                    self.LE_NombreArchivo.setEnabled(False)
                    self.label.setEnabled(False)
                    self.LE_Extension.setEnabled(False)
                    self.pB_Cargar.setEnabled(False)
                if Extension == ".xls":
                    File = NombreArchivo + Extension
                    Data = pd.read_excel(File, header = None)
                    DataStr = str(Data)
                    self.txt_Texto.setText(DataStr)
                    self.label_19.setHidden(False) #Se activan iconos de que se leyo correctamente el archivo
                    self.cb_Algoritmo.setEnabled(True)
                    self.lbl_Algoritmo.setEnabled(True)
                    self.pB_Confirmar.setEnabled(True)
                    self.lbl_SolicitoArchivo.setEnabled(False)
                    self.LE_NombreArchivo.setEnabled(False)
                    self.label.setEnabled(False)
                    self.LE_Extension.setEnabled(False)
                    self.pB_Cargar.setEnabled(False)
            except:
                #Mensaje de error en caso de no encontrar el archivo
                QMessageBox.critical(self, "ERROR", "No se encontro el archivo")

    def Algoritmos(self): #Funcion que controla la QComboBox para la seleccion de los distintos algoritos implementados
        AlgoritmoSeleccionado = str(self.cb_Algoritmo.currentText())
        print(AlgoritmoSeleccionado)

        if AlgoritmoSeleccionado == "Apriori":
            #Se habilitan todas las entradas
            self.label_2.setEnabled(True)
            self.label_3.setEnabled(True)
            self.label_4.setEnabled(True)
            self.label_5.setEnabled(True)
            self.label_7.setEnabled(True)
            self.LE_Confianza.setEnabled(True)
            self.LE_Soporte.setEnabled(True)
            self.LE_Tamanio.setEnabled(True)
            self.LE_lift.setEnabled(True)
            self.pB_Enviar.setEnabled(True)
            self.pB_Nuevo.setEnabled(True)
            global x,y
            x = len(Data.index)
            y = len(Data.columns)
            print(x,y)
            # Se determinan las transacciones realizadas
            global Transacciones
            for i in range(0, int(x)):
                Transacciones.append([str(Data.values[i, j]) for j in range(0, int(y))])
            print(Transacciones) #Para ver las transacciones ya que tardan en consola
        if AlgoritmoSeleccionado == "Correlacional":
            self.label_2.setEnabled(False)
            self.label_3.setEnabled(False)
            self.label_4.setEnabled(False)
            self.label_5.setEnabled(False)
            self.label_7.setEnabled(False)
            self.LE_Confianza.setEnabled(False)
            self.LE_Soporte.setEnabled(False)
            self.LE_Tamanio.setEnabled(False)
            self.LE_lift.setEnabled(False)
            self.pB_Enviar.setEnabled(False)
            self.label_6.setEnabled(True)
            self.label_8.setEnabled(True)
            self.label_9.setEnabled(True)
            self.pB_Enviar_2.setEnabled(True)
            self.pB_Nuevo.setEnabled(True)
            Matriz = Data.corr(method='pearson')
            self.txt_Texto_2.insertPlainText(str(Matriz) + "\n")
            plt.matshow(Matriz)
            plt.show()
        if AlgoritmoSeleccionado == "Metricas de Similitud":
            self.pB_ok.setEnabled(True)
            self.cb_Algoritmo_2.setEnabled(True)
            self.label_10.setEnabled(True)
            self.pB_ok.clicked.connect(self.OptMetricas)

        if AlgoritmoSeleccionado == "Clustering Particional":
            self.pB_Nuevo.setEnabled(True)
            self.pB_Enviar_4.setEnabled(True)

            #Obtencion del nombre del archivo
            NombreArchivo = self.LE_NombreArchivo.text()
            #Obtencion de la extensión del archivo
            Extension = self.LE_Extension.text()
            Extension.lower
            global DataCluster
            if Extension == ".csv":
                File = NombreArchivo + Extension
                DataCluster = pd.read_csv(File)
                DataStr = str(Data)
                self.txt_Texto.setText(DataStr)
                self.label_19.setHidden(False) #Se activan iconos de que se leyo correctamente el archivo
                self.cb_Algoritmo.setEnabled(True)
                self.lbl_Algoritmo.setEnabled(True)
                self.pB_Confirmar.setEnabled(True)
                self.lbl_SolicitoArchivo.setEnabled(False)
                self.LE_NombreArchivo.setEnabled(False)
                self.label.setEnabled(False)
                self.LE_Extension.setEnabled(False)
                self.pB_Cargar.setEnabled(False)
                for column in DataCluster:
                    self.LW_SelecVar.addItem(column)
            if Extension == ".xls":
                File = NombreArchivo + Extension
                DataCluster = pd.read_excel(File)
                DataStr = str(Data)
                self.txt_Texto.setText(DataStr)
                self.label_19.setHidden(False) #Se activan iconos de que se leyo correctamente el archivo
                self.cb_Algoritmo.setEnabled(True)
                self.lbl_Algoritmo.setEnabled(True)
                self.pB_Confirmar.setEnabled(True)
                self.lbl_SolicitoArchivo.setEnabled(False)
                self.LE_NombreArchivo.setEnabled(False)
                self.label.setEnabled(False)
                self.LE_Extension.setEnabled(False)
                self.pB_Cargar.setEnabled(False)



    def LeerParametrosApriori(self): #Funcion de lectura de los parametros necesarios para poder ejecutar el algoritmo apriorio
        #Se obtienen los parametros de Apriori probenientes de las cajas de texto y sea castean a flotantes
        #En caso de generar un error se muestra un error
        try:
            global Confianza, Soporte, Lift, Tamanio
            Confianza = float(self.LE_Confianza.text())
            Soporte = float(self.LE_Soporte.text())
            Lift = float(self.LE_lift.text())
            Tamanio = float(self.LE_Tamanio.text())
        except:
            QMessageBox.critical(self, "ERROR", "No fue posible realizar ningun calculo, verifique sus datos.")

        #Se generan todas la posiblres reglar con los datos introducidos por el usuario
        Reglas = apriori(Transacciones, min_support=Soporte, min_confidence=Confianza, min_lift=Lift, min_length=Tamanio)
        Resultados = list(Reglas)

        #Se muestra el numero de reglas obtenidas en el area de texto junto con cada una de las reglas obtenidas
        self.txt_Texto_2.insertPlainText("El total de reglas son:" + str(len(Resultados)) +"\n")
        for item in Resultados:
            pair = item[0]
            items = [x for x in pair]
            self.txt_Texto_2.insertPlainText("Regla: " + items[0] + " -> " + items[1] + "\n")
            self.txt_Texto_2.insertPlainText("Soporte: " + str(item[1]) + "\n")
            self.txt_Texto_2.insertPlainText("Confianza: " + str(item[2][0][2]) + "\n")
            self.txt_Texto_2.insertPlainText("Lift: " + str(item[2][0][3]) + "\n")
            self.txt_Texto_2.insertPlainText("=========================================================================" + "\n")

    def LeerParametrosCorrelacional(self): #Funcion de lectura de parametros para el metodo de correlaciones donde de obtiene la grafica de dichos datos
        Parametro1 = self.LE_Parametro.text()
        Parametro2 = self.LE_Parametro_2.text()
        try:
            plt.close() #Limpiar pantalla grafico
            plt.plot(Data[Parametro1], Data[Parametro2], 'r+')
            plt.ylabel(Parametro2)
            plt.xlabel(Parametro1)
            plt.show()
        except:
            QMessageBox.critical(self, "ERROR", "El parametro no existe, no es posible graficar")

    def OptMetricas(self):
        if(self.cb_Algoritmo_2.currentText() == "Elementos"):
            self.label_22.setEnabled(True)
            self.label_23.setEnabled(True)
            self.LE_Elemento_1.setEnabled(True)
            self.LE_Elemento_2.setEnabled(True)
            self.pB_Euclidiana.setEnabled(True)
            self.pB_Chebyshev.setEnabled(True)
            self.pB_Manhattan.setEnabled(True)
            self.pB_Minkowsky.setEnabled(True)
            self.pB_Nuevo.setEnabled(True)
        else:
            self.label_22.setEnabled(False)
            self.label_23.setEnabled(False)
            self.LE_Elemento_1.setEnabled(False)
            self.LE_Elemento_2.setEnabled(False)
            self.pB_Euclidiana.setEnabled(True)
            self.pB_Chebyshev.setEnabled(True)
            self.pB_Manhattan.setEnabled(True)
            self.pB_Minkowsky.setEnabled(True)
            self.pB_Nuevo.setEnabled(True)

    def Euclidiana(self):
            DataMetricas = 0
            DataMetricas = Data
            DataMetricas.drop(0)
            x = len(DataMetricas.index)
            y = len(DataMetricas.columns)

            DataCalc = {}
            for i in range(0,y):
                Lista = []
                for j in range(0,y):
                    Lista.append('--')
                DataCalc[i] = Lista
            MatrizSimilitudes = pd.DataFrame(data = DataCalc)
            for i in range(0, y):
                for j in range(0, i+1):
                    E1 = DataMetricas.iloc[i]
                    E2 = DataMetricas.iloc[j]
                    Temp1 = []
                    Temp2 = []
                    for k in range(0, len(E1)):
                        try:
                            Temp1.append(float(E1[k]))
                        except ValueError:
                            pass
                    for k in range(0, len(E2)):
                        try:
                            Temp2.append(float(E2[k]))
                        except ValueError:
                            pass
                    dist = "{0:.05f}".format(math.sqrt(sum((Temp1-Temp2)**2 for Temp1, Temp2 in zip(Temp1, Temp2))))
                    MatrizSimilitudes.iloc[i, j] = dist
            MatrizSimilitudes.drop(0)
            print(MatrizSimilitudes)


            if(self.cb_Algoritmo_2.currentText() == 'Elementos'):
                try:
                    val1 = int(self.LE_Elemento_1.text()) - 1
                    val2 = int(self.LE_Elemento_2.text()) - 1 #el mas 1 es por la forma de los indices de la Matriz
                    self.txt_Texto_2.insertPlainText("La distancia Euclidiana entre los elementos " + str(val1+1) + " y " + str(val2+1) + " es:" + MatrizSimilitudes[val1][val2] + "\n")
                    self.txt_Texto_2.insertPlainText("=========================================================================" + "\n")
                except:
                    QMessageBox.critical(self, "ERROR", "Verifique los datos introducidos")
            else:
                self.txt_Texto_2.insertPlainText(str(MatrizSimilitudes))

    def Manhattan(self):
        DataMetricas = 0
        DataMetricas = Data
        DataMetricas.drop(0)
        x = len(DataMetricas.index)
        y = len(DataMetricas.columns)

        DataCalc = {}
        for i in range(0,y):
            Lista = []
            for j in range(0,y):
                Lista.append('--')
            DataCalc[i] = Lista
        MatrizSimilitudes = pd.DataFrame(data = DataCalc)
        for i in range(0, y):
            for j in range(0, i+1):
                E1 = DataMetricas.iloc[i]
                E2 = DataMetricas.iloc[j]
                Temp1 = []
                Temp2 = []
                for k in range(0, len(E1)):
                    try:
                        Temp1.append(float(E1[k]))
                    except ValueError:
                        pass
                for k in range(0, len(E2)):
                    try:
                        Temp2.append(float(E2[k]))
                    except ValueError:
                        pass
                dist = "{0:.5f}".format(distance.cityblock(Temp1, Temp2))
                MatrizSimilitudes.iloc[i, j] = dist

        MatrizSimilitudes.drop(0)
        print(MatrizSimilitudes)
        if(self.cb_Algoritmo_2.currentText() == 'Elementos'):
            try:
                val1 = int(self.LE_Elemento_1.text()) - 1
                val2 = int(self.LE_Elemento_2.text()) - 1 #el mas 1 es por la forma de los indices de la Matriz
                self.txt_Texto_2.insertPlainText("La distancia Manhattan entre los elementos " + str(val1+1) + " y " + str(val2+1) + " es:" + MatrizSimilitudes[val1][val2] + "\n")
                self.txt_Texto_2.insertPlainText("=========================================================================" + "\n")
            except:
                QMessageBox.critical(self, "ERROR", "Verifique los datos introducidos")
        else:
            self.txt_Texto_2.insertPlainText(str(MatrizSimilitudes))

    def Minkowsky(self):
        DataMetricas = 0
        DataMetricas = Data
        DataMetricas.drop(0)
        x = len(DataMetricas.index)
        y = len(DataMetricas.columns)

        DataCalc = {}
        for i in range(0,y):
            Lista = []
            for j in range(0,y):
                Lista.append('--')
            DataCalc[i] = Lista
        MatrizSimilitudes = pd.DataFrame(data = DataCalc)
        for i in range(0, y):
            for j in range(0, i+1):
                E1 = DataMetricas.iloc[i]
                E2 = DataMetricas.iloc[j]
                Temp1 = []
                Temp2 = []
                for k in range(0, len(E1)):
                    try:
                        Temp1.append(float(E1[k]))
                    except ValueError:
                        pass
                for k in range(0, len(E2)):
                    try:
                        Temp2.append(float(E2[k]))
                    except ValueError:
                        pass
                dist = "{0:.5f}".format(distance.minkowski(Temp1, Temp2))
                MatrizSimilitudes.iloc[i, j] = dist

        MatrizSimilitudes.drop(0)
        print(MatrizSimilitudes)
        if(self.cb_Algoritmo_2.currentText() == 'Elementos'):
            try:
                val1 = int(self.LE_Elemento_1.text()) - 1
                val2 = int(self.LE_Elemento_2.text()) - 1 #el mas 1 es por la forma de los indices de la Matriz
                self.txt_Texto_2.insertPlainText("La distancia Minkowsky entre los elementos " + str(val1+1) + " y " + str(val2+1) + " es:" + MatrizSimilitudes[val1][val2] + "\n")
                self.txt_Texto_2.insertPlainText("=========================================================================" + "\n")
            except:
                QMessageBox.critical(self, "ERROR", "Verifique los datos introducidos")
        else:
            self.txt_Texto_2.insertPlainText(str(MatrizSimilitudes))

    def Chebyshev(self):
        DataMetricas = 0
        DataMetricas = Data
        DataMetricas.drop(0)
        x = len(DataMetricas.index)
        y = len(DataMetricas.columns)

        DataCalc = {}
        for i in range(0,y):
            Lista = []
            for j in range(0,y):
                Lista.append('--')
            DataCalc[i] = Lista
        MatrizSimilitudes = pd.DataFrame(data = DataCalc)
        for i in range(0, y):
            for j in range(0, i+1):
                E1 = DataMetricas.iloc[i]
                E2 = DataMetricas.iloc[j]
                Temp1 = []
                Temp2 = []
                for k in range(0, len(E1)):
                    try:
                        Temp1.append(float(E1[k]))
                    except ValueError:
                        pass
                for k in range(0, len(E2)):
                    try:
                        Temp2.append(float(E2[k]))
                    except ValueError:
                        pass
                dist = "{0:.5f}".format(distance.chebyshev(Temp1, Temp2))
                MatrizSimilitudes.iloc[i, j] = dist

        MatrizSimilitudes.drop(0)
        if(self.cb_Algoritmo_2.currentText() == 'Elementos'):
            try:
                val1 = int(self.LE_Elemento_1.text()) - 1
                val2 = int(self.LE_Elemento_2.text()) - 1 #el mas 1 es por la forma de los indices de la Matriz
                self.txt_Texto_2.insertPlainText("La distancia de Chebyshev entre lso elementos " + str(val1+1) + " y " + str(val2+1) + " es:" + MatrizSimilitudes[val1][val2] + "\n")
                self.txt_Texto_2.insertPlainText("=========================================================================" + "\n")
            except:
                QMessageBox.critical(self, "ERROR", "Verifique los datos introducidos")
        else:
            self.txt_Texto_2.insertPlainText(str(MatrizSimilitudes))

    def ClusteringParticional(self):

        VarSelec = self.LW_SelecVar.selectedItems()
        select = []
        row =[]
        for x in range(len(VarSelec)):
            select.append(self.LW_SelecVar.selectedItems()[x].text())
        print(select)

        Matriz = DataCluster.corr(method='pearson')
        VariablesModelo = np.array(DataCluster[select])
        print(VariablesModelo)

        SSE = []
        for i in range(2, 16):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(VariablesModelo)
            SSE.append(km.inertia_)

        #Se grafica SSE en función de k
        plt.figure(figsize=(7, 3))
        plt.plot(range(2, 16), SSE, marker='o')
        plt.xlabel('Cantidad de clusters *k*')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.show()

        #Se crean los clusters
        #random_state se utiliza para inicializar el generador interno de números aleatorios (mismo resultado)
        MParticional = KMeans(n_clusters=5, random_state=0).fit(VariablesModelo)
        MParticional.predict(VariablesModelo)

        DataCluster['clusterP'] = MParticional.labels_
        self.txt_Texto_2.insertPlainText("Los Clusters son:\n")
        self.txt_Texto_2.insertPlainText(str(DataCluster.groupby(['clusterP'])['clusterP'].count()) + "\n")
        self.txt_Texto_2.insertPlainText("=========================================================================" + "\n")

        CentroidesP = MParticional.cluster_centers_

        self.txt_Texto_2.insertPlainText("La matriz de centroides es:\n")
        self.txt_Texto_2.insertPlainText(str(pd.DataFrame(CentroidesP.round(4))) + "\n")

        # Gráfica de los elementos y los centros de los clusters
        plt.rcParams['figure.figsize'] = (7, 3)
        plt.style.use('ggplot')
        colores=['red', 'blue', 'cyan', 'green', 'yellow']
        asignar=[]
        for row in MParticional.labels_:
            asignar.append(colores[row])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter (VariablesModelo[:, 0], VariablesModelo[:, 1], VariablesModelo[:, 2], marker='o', c=asignar, s=60)
        ax.scatter(CentroidesP[:, 0], CentroidesP[:, 1], CentroidesP[:, 2], marker='*', c=colores, s=1000)
        plt.show()

    def LeerParametrosDiagnostico(self): #Funcion que implementa la lectura y ejecucion del diagnostico con los datos propocionados
        global Texture, Area, Compactness, Concavity, FractalDim, Symmetry
        try:
            Texture = float(self.LE_Texture.text())
            Area = float(self.LE_Area.text())
            Compactness = float(self.LE_Compactness.text())
            Concavity = float(self.LE_Concavity.text())
            FractalDim = float(self.LE_FractalDim.text())
            Symmetry = float(self.LE_Symmetry.text())
        except:
            QMessageBox.critical(self, "ERROR", "Verifique los datos introducidos")

        ModeloDeClasificacion = 11.72-0.19*Texture-0.01*Area-2.27*Compactness-3.08*Concavity-0.88*Symmetry-0.21*FractalDim
        Probabilidad = 1/(1+np.exp(ModeloDeClasificacion))

        if(Probabilidad < 0.5):
            self.label_13.setHidden(False)
            self.LE_Texture.setEnabled(False)
            self.LE_Area.setEnabled(False)
            self.LE_Compactness.setEnabled(False)
            self.LE_Symmetry.setEnabled(False)
            self.LE_FractalDim.setEnabled(False)
            self.LE_Concavity.setEnabled(False)
            self.lb_Diagnostico.setText('Tumor benigno')
        else:
            self.label_12.setHidden(False)
            self.lb_Diagnostico.setText('Tumor Maligno')
            self.LE_Texture.setEnabled(False)
            self.LE_Area.setEnabled(False)
            self.LE_Compactness.setEnabled(False)
            self.LE_Symmetry.setEnabled(False)
            self.LE_FractalDim.setEnabled(False)
            self.LE_Concavity.setEnabled(False)

    def NuevoAnalisis(self): #Funcion encargada de reinician un analisis perimitiendo seleccionar un nuevo algoritmo con el que se desea trabajar
        self.pB_Confirmar.setEnabled(False)
        self.lbl_SolicitoArchivo.setEnabled(True)
        self.LE_NombreArchivo.setEnabled(True)
        self.label.setEnabled(True)
        self.LE_Extension.setEnabled(True)
        self.pB_Cargar.setEnabled(True)

        self.label_2.setEnabled(False)
        self.label_3.setEnabled(False)
        self.label_4.setEnabled(False)
        self.label_5.setEnabled(False)
        self.label_6.setEnabled(False)
        self.label_7.setEnabled(False)
        self.label_8.setEnabled(False)
        self.label_9.setEnabled(False)
        self.label_10.setEnabled(False)
        self.label_11.setEnabled(False)
        self.label_12.setHidden(True)
        self.label_13.setHidden(True)
        self.label_14.setEnabled(False)
        self.label_15.setEnabled(False)
        self.label_16.setEnabled(False)
        self.label_17.setEnabled(False)
        self.label_18.setEnabled(False)
        self.label_19.setHidden(True)
        self.label_20.setEnabled(False)
        self.label_22.setEnabled(False)
        self.label_23.setEnabled(False)
        self.lbl_Algoritmo.setEnabled(False)
        self.pB_Enviar.setEnabled(False)
        self.pB_Enviar_2.setEnabled(False)
        self.pB_Euclidiana.setEnabled(False)
        self.pB_Chebyshev.setEnabled(False)
        self.pB_Manhattan.setEnabled(False)
        self.pB_Minkowsky.setEnabled(False)
        self.pB_Enviar_4.setEnabled(False)
        self.pB_Confirmar.setEnabled(False)
        self.cb_Algoritmo.setEnabled(False)
        self.LE_Confianza.setEnabled(False)
        self.LE_Soporte.setEnabled(False)
        self.LE_Tamanio.setEnabled(False)
        self.LE_lift.setEnabled(False)
        self.lbl_Titulo_2.setEnabled(False)
        self.LE_Texture.setEnabled(False)
        self.LE_Compactness.setEnabled(False)
        self.LE_Symmetry.setEnabled(False)
        self.LE_FractalDim.setEnabled(False)
        self.LE_Concavity.setEnabled(False)
        self.pB_Enviar_5.setEnabled(False)
        self.lb_Diagnostico.setEnabled(False)
        self.pB_Nuevo.setEnabled(False)
        self.LW_SelecVar.setEnabled(False)
        self.txt_Texto_2.clear()
        self.txt_Texto.clear()
        self.LE_NombreArchivo.clear()
        self.LE_Extension.clear()
        self.LE_Confianza.clear()
        self.LE_Soporte.clear()
        self.LE_Tamanio.clear()
        self.LE_lift.clear()
        self.LE_Parametro.clear()
        self.LE_Parametro_2.clear()
        self.LE_Elemento_1.clear()
        self.LE_Elemento_2.clear()

        plt.close()
        Data = 0
        x = 0
        y = 0
        Transacciones = []
        Confianza = 0
        Soporte = 0
        Lift = 0
        Tamanio = 0
        AuxCambio = 0
        Texture = 0
        Area = 0
        Compactness = 0
        Concavity = 0
        FractalDim = 0
        Symmetry = 0
        DataCluster = 0

    def NuevoDiagnostico(self): #Funcion encargada de habiliar denuevo las entrada para hacer un Diagnostico
        self.LE_Texture.setEnabled(True)
        self.LE_Area.setEnabled(True)
        self.LE_Compactness.setEnabled(True)
        self.LE_Symmetry.setEnabled(True)
        self.LE_FractalDim.setEnabled(True)
        self.LE_Concavity.setEnabled(True)
        self.label_12.setHidden(True)
        self.label_13.setHidden(True)
        self.lb_Diagnostico.setText('')
        self.LE_Texture.clear()
        self.LE_Area.clear()
        self.LE_Compactness.clear()
        self.LE_Symmetry.clear()
        self.LE_FractalDim.clear()
        self.LE_Concavity.clear()

    def Cambiar(self): #Funcion encargada de prender y apargar el Analizador como el area de Diagnostico de forma alternada.
        global AuxCambio
        if(AuxCambio == 0):
            AuxCambio = 1
            #Se apagan toda el area de Analizador Datos
            self.pB_Confirmar.setEnabled(False)
            self.lbl_SolicitoArchivo.setEnabled(False)
            self.LE_NombreArchivo.setEnabled(False)
            self.label.setEnabled(False)
            self.LE_Extension.setEnabled(False)
            self.pB_Cargar.setEnabled(False)
            self.label_2.setEnabled(False)
            self.label_3.setEnabled(False)
            self.label_4.setEnabled(False)
            self.label_5.setEnabled(False)
            self.label_6.setEnabled(False)
            self.label_7.setEnabled(False)
            self.label_8.setEnabled(False)
            self.label_9.setEnabled(False)
            self.label_10.setEnabled(False)
            self.label_11.setEnabled(False)
            self.label_12.setHidden(True)
            self.label_13.setHidden(True)
            self.label_14.setEnabled(False)
            self.label_15.setEnabled(False)
            self.label_16.setEnabled(False)
            self.label_17.setEnabled(False)
            self.label_18.setEnabled(False)
            self.label_19.setHidden(True)
            self.label_20.setEnabled(False)
            self.label_22.setEnabled(False)
            self.label_23.setEnabled(False)
            self.lbl_Algoritmo.setEnabled(False)
            self.LE_Elemento_1.setEnabled(False)
            self.LE_Elemento_2.setEnabled(False)
            self.pB_Enviar.setEnabled(False)
            self.pB_Enviar_2.setEnabled(False)
            self.pB_Euclidiana.setEnabled(False)
            self.pB_Chebyshev.setEnabled(False)
            self.pB_Manhattan.setEnabled(False)
            self.pB_Minkowsky.setEnabled(False)
            self.pB_Enviar_4.setEnabled(False)
            self.pB_Confirmar.setEnabled(False)
            self.cb_Algoritmo.setEnabled(False)
            self.LE_Confianza.setEnabled(False)
            self.LE_Soporte.setEnabled(False)
            self.LE_Tamanio.setEnabled(False)
            self.LE_lift.setEnabled(False)
            self.lbl_Titulo_2.setEnabled(False)
            self.LE_Texture.setEnabled(False)
            self.LE_Compactness.setEnabled(False)
            self.LE_Symmetry.setEnabled(False)
            self.LE_FractalDim.setEnabled(False)
            self.LE_Concavity.setEnabled(False)
            self.pB_Enviar_5.setEnabled(False)
            self.lb_Diagnostico.setEnabled(False)
            self.pB_Nuevo.setEnabled(False)

            #PrenderDiag
            self.lbl_Titulo_2.setEnabled(True)
            self.label_14.setEnabled(True)
            self.label_15.setEnabled(True)
            self.label_16.setEnabled(True)
            self.label_17.setEnabled(True)
            self.label_18.setEnabled(True)
            self.label_19.setHidden(True)
            self.label_20.setEnabled(True)
            self.LE_Texture.setEnabled(True)
            self.LE_Area.setEnabled(True)
            self.LE_Compactness.setEnabled(True)
            self.LE_Concavity.setEnabled(True)
            self.LE_FractalDim.setEnabled(True)
            self.LE_Symmetry.setEnabled(True)
            self.pB_Enviar_5.setEnabled(True)
            self.lb_Diagnostico.setEnabled(True)
            self.pB_Nuevo_2.setEnabled(True)
        else:
            AuxCambio = 0
            #ApagarDiag
            self.lbl_Titulo_2.setEnabled(False)
            self.label_14.setEnabled(False)
            self.label_15.setEnabled(False)
            self.label_16.setEnabled(False)
            self.label_17.setEnabled(False)
            self.label_18.setEnabled(False)
            self.label_19.setHidden(False)
            self.label_20.setEnabled(False)
            self.LE_Texture.setEnabled(False)
            self.LE_Area.setEnabled(False)
            self.LE_Compactness.setEnabled(False)
            self.LE_Concavity.setEnabled(False)
            self.LE_FractalDim.setEnabled(False)
            self.LE_Symmetry.setEnabled(False)
            self.pB_Enviar_5.setEnabled(False)
            self.lb_Diagnostico.setEnabled(False)
            self.pB_Nuevo_2.setEnabled(False)
            #PrenderAnalizador
            self.lbl_SolicitoArchivo.setEnabled(True)
            self.LE_NombreArchivo.setEnabled(True)
            self.label.setEnabled(True)
            self.LE_Extension.setEnabled(True)
            self.pB_Cargar.setEnabled(True)
            self.label_19.setHidden(True)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ventana = AnalizadorDataApp()
    ventana.show()
    sys.exit(app.exec_())
