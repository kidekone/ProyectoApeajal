#Librerias utilizadas en el proyecto
from operator import imod
import os
from django.shortcuts import redirect, render
import pandas as pd
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
import statsmodels.formula.api as smf
import numpy as np
import tensorflow as tf
from django.contrib import messages

def index(request):
    if request.user.is_authenticated:
        datos=1
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")
        df = pd.DataFrame({"Cantidad":datos["Cantidad"], "Empacadora":datos["Empacadora"],})
            
        df = df.groupby(["Empacadora"], as_index=False)['Cantidad'].sum()
        df = pd.DataFrame(df)
            
        df = pd.DataFrame({"Empacadora":df['Empacadora']})
        df = df['Empacadora'].tolist()
        
        datosP = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")
        dfP = pd.DataFrame({"Cantidad":datosP["Cantidad"], "Pais":datosP["País Destino"],})        
        dfP = dfP.groupby(["Pais"], as_index=False)['Cantidad'].sum()
        dfP = pd.DataFrame(dfP)
        dfP = pd.DataFrame({"Pais":dfP['Pais']})
        dfP = dfP['Pais'].tolist()
        return render(request, 'index.html',{'df': df,'dfP': dfP})
        
    else:
        return redirect("/login/")

def logout_request(request):
    logout(request)
    return redirect("/login/")

def login_request(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            usuario = form.cleaned_data.get('username')
            contraseña = form.cleaned_data.get('password')
            user = authenticate(username=usuario, password=contraseña)

            if user is not None:
                login(request, user)
                return redirect("index/")
            else:
                messages.error(request, "usuario equivocado")
        else:
            messages.error(request, "usuario equivocado")

    form = AuthenticationForm()
    return render(request, "login.html")

def destinos(request):
    if request.user.is_authenticated:
        return render(request, "destinos.html")
    else:
        return redirect("/login/")

def your_handler(request):
    arr = [1, 2, 3, 4]
    return render(request, 'index.html', {'arr':arr})

def proPaises(request):
    if request.user.is_authenticated:
        return render(request, "pronosticoPais.html")
    else:
        return redirect("/login/")

def proGeneralPaises(request):
    if request.user.is_authenticated:
        return render(request, "pronosticoGeneralPaises.html")
    else:
        return redirect("/login/")

def proEmpacadora(request):
    if request.user.is_authenticated:
        return render(request, "pronosticoEmpacadora.html")
    else:
        return redirect("/login/")

def empacadoraG(request):
    if request.user.is_authenticated:
        return render(request, "empacadoraGeneral.html")
    else:
        return redirect("/login/")

def empacadoraI(request):
    if request.user.is_authenticated:
        return render(request, "empacadoraIndividual.html")
    else:
        return redirect("/login/")

def paises(request):
    if request.user.is_authenticated:
        return render(request, "paises.html")
    else:
        return redirect("/login/")

def listaPaises():
    datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")
    df = pd.DataFrame({"Cantidad":datos["Cantidad"], "Pais":datos["País Destino"],})        
    df = df.groupby(["Pais"], as_index=False)['Cantidad'].sum()
    df = pd.DataFrame(df)
    df = pd.DataFrame({"Pais":df['Pais']})
    dfPais = df['Pais'].tolist()
    return render('index.html',{'dfPais': dfPais})

# GRÁFICAS
# Grafica de empacadora individual por temporada
# --------------------------------------------------------------------------------------------------
def empacadorasIndividual(request):
    if request.user.is_authenticated:
        plt.close()
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")
        df = pd.DataFrame({"Empacadora":datos["Empacadora"], 'Fecha':datos["Fecha expedición"], "Variedad":datos["Variedad"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"], "idGrupo":datos["Id Predicción"]})
        df.loc[(df.Unidad=="Kilogramos"),'Cantidad']=(df['Cantidad']/1000)
        df.loc[(df.Unidad=="Kilogramos"),'Unidad']="Toneladas"

        year = request.POST.get("year")
        date1p1=year+'-06-01'
        date2p1=year+'-11-30'

        date1p2=year+'-12-01'
        yearAux = int(year)
        yearAux = yearAux+1
        year2 = str(yearAux)
        date2p2=year2+'-05-31'

        v_empacadora = request.POST.get("empacadora")
        
        periodo1 =df.loc[df["Fecha"].between(date1p1, date2p1) &(df["Variedad"]=="Méndez")]
        periodo1 = pd.DataFrame(periodo1)

        periodo1 = periodo1.groupby('Empacadora',as_index=False)["Cantidad"].sum()
        periodo1 = pd.DataFrame(periodo1)
        user1 = periodo1[periodo1['Empacadora'] == v_empacadora] 
        user1 = pd.DataFrame(user1)

        periodo2 =df.loc[df["Fecha"].between(date1p2, date2p2) &(df["Variedad"]=="Hass")]
        periodo2 = pd.DataFrame(periodo2)
        periodo2 = periodo2.groupby('Empacadora',as_index=False)["Cantidad"].sum()
        periodo2 = pd.DataFrame(periodo2)
        user2 = periodo2[periodo2['Empacadora'] == v_empacadora] 
        user2 = pd.DataFrame(user2)

        pd.options.display.float_format = '{:,.2f}'.format
        tabla = pd.DataFrame({"Cantidad Méndez":user1['Cantidad'],"Cantidad Hass":user2['Cantidad']})
        tabla.set_index(['Cantidad Méndez','Cantidad Hass'], inplace = True)

        valores = user1['Cantidad']
        valores1 = user2['Cantidad']
        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')
        fig = plt.figure(figsize =(12,6))
        plt.barh('Hass',valores1,0.3,label="Periodo Dic- May Tipo: Hass",color="green")
        plt.barh('Méndez',valores,0.3,label="Periodo Jun- Nov Tipo: Méndez",color="#FDBD31")

        plt.ylabel('Tipo')
        plt.xlabel('Toneladas exportadas')
        plt.title("Gráfica de exportacion de la empacadora: "+v_empacadora)
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)
        plt.legend()
        fig.savefig('static/imagenes/empacadora_Invidivual.png')
        
        tabla = tabla.to_html(classes='table table-striped', col_space=3)
        context = {'tabla': tabla}  # change
        return render(request, 'empacadoraIndividual.html', context)
    else:
        return redirect("/login/")


# Grafica de todas las empacadoras por temporada
# --------------------------------------------------------------------------------------------------
def empacadorasGeneral(request):
    if request.user.is_authenticated:
        plt.close()
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")
        df = pd.DataFrame({"Empacadora":datos["Empacadora"], 'Fecha':datos["Fecha expedición"], "Variedad":datos["Variedad"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"], "idGrupo":datos["Id Predicción"]})
        df.loc[(df.Unidad=="Kilogramos"),'Cantidad']=(df['Cantidad']/1000)
        df.loc[(df.Unidad=="Kilogramos"),'Unidad']="Toneladas"
        
        year = request.POST.get("year")
        date1p1=year+'-06-01'
        date2p1=year+'-11-30'
        date1p2=year+'-12-01'
        yearAux = int(year)
        yearAux = yearAux+1
        year2 = str(yearAux)
        date2p2=year2+'-05-31'

        # Código para extraer datos en una fecha para el periodo uno y asignar los datos filtrados a un dataframe nuevo
        periodo1 =df.loc[df["Fecha"].between(date1p1, date2p1) &(df["Variedad"]=="Méndez")]
        periodo1 = pd.DataFrame(periodo1)
        periodo1 = periodo1.groupby('Empacadora',as_index=False)["Cantidad"].sum()
        periodo1 = pd.DataFrame(periodo1)
        
        # Código para extraer datos en una fecha para el periodo dos y asignar los datos filtrados a un dataframe nuevo
        periodo2 =df.loc[df["Fecha"].between(date1p2, date2p2) &(df["Variedad"]=="Hass")]
        periodo2 = pd.DataFrame(periodo2)
        periodo2 = periodo2.groupby('Empacadora',as_index=False)["Cantidad"].sum()
        periodo2 = pd.DataFrame(periodo2)

        empacadoras = df.groupby('Empacadora',as_index=False)["Cantidad"].sum()

        pd.options.display.float_format = '{:,.2f}'.format
        tabla = pd.DataFrame({"Empacadora":empacadoras.Empacadora,"Cantidad Mendéz":periodo1.Cantidad,"Cantidad Hass":periodo2.Cantidad})
        tabla.set_index('Empacadora', inplace = True)
        
        # Graficando los dos periodos de fechas
        valores = periodo1['Cantidad']
        valores1 = periodo2['Cantidad']

        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')

        fig = plt.figure(figsize =(12,6))
        plt.barh(periodo1['Empacadora'], valores, 0.3, align='edge',label="Periodo Jun- Nov Tipo: Méndez", color="darkgreen")
        plt.barh(periodo2['Empacadora'], valores1, 0.3, align='center',label="Periodo Dic- May Tipo: Hass", color="darkorange")
        plt.xlabel('Toneladas Exportadas')
        plt.ylabel('Empacadoras')
        plt.title('Empacadoras en general')
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)
        plt.legend()
        plt.subplots_adjust(left=0.300, bottom=0.16, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
        #plt.show()
        
        fig.savefig('static/imagenes/empacadora_General.png')
        
        tabla = tabla.to_html(classes='table table-striped', table_id="empacadoras")
        context = {'tabla': tabla}  # change

        return render(request, 'empacadoraGeneral.html', context)
    else:
        return redirect("/login/")

# Grafica de los destinos importantes por temporada
# --------------------------------------------------------------------------------------------------
def destinosTemporada(request):
    if request.user.is_authenticated:
        plt.close()
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")
        df = pd.DataFrame({'Fecha':datos["Fecha expedición"], "Variedad":datos["Variedad"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"],"idContinente":datos["Id Continente"]})

        df.loc[(df.Unidad=="Kilogramos"),'Cantidad']=(df['Cantidad']/1000)
        df.loc[(df.Unidad=="Kilogramos"),'Unidad']="Toneladas"

        year = request.POST.get("year")
        date1p1=year+'-06-01'
        date2p1=year+'-11-30'
        date1p2=year+'-12-01'
        yearAux = int(year)
        yearAux = yearAux+1
        year2 = str(yearAux)
        date2p2=year2+'-05-31'

        periodo1 =df.loc[df["Fecha"].between(date1p1, date2p1) &(df["Variedad"]=="Méndez")]
        periodo1 = pd.DataFrame(periodo1)
        periodo1 = periodo1.groupby(["idContinente"], as_index=False)['Cantidad'].sum()
        periodo1 = pd.DataFrame(periodo1)

        periodo2 =df.loc[df["Fecha"].between(date1p2, date2p2) &(df["Variedad"]=="Hass")]
        periodo2 = pd.DataFrame(periodo2)
        periodo2 = df.groupby(["idContinente"], as_index=False)['Cantidad'].sum()
        periodo2 = pd.DataFrame(periodo2)

        continentes = ["Medio Oriente","América","Europa","Asia"]
        pd.options.display.float_format = '{:,.2f}'.format
        tabla = pd.DataFrame({"continentes":continentes,"Cantidad Mendéz":periodo1.Cantidad, "Cantidad Hass":periodo2.Cantidad})
        tabla.set_index('continentes', inplace = True)
        valores = periodo1['Cantidad']
        valores1 = periodo2['Cantidad']
        continentes = ["Medio Oriente","América","Europa","Asia"]
        x = np.arange(len(continentes))
    
        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')
        x = np.arange(len(continentes))
        
        fig = plt.figure(figsize =(12,6))
        
        
        plt.bar(periodo2['idContinente']+0.00,valores1,0.2,label="Periodo Dic- May Tipo: Hass",color="green")
        plt.bar(periodo1['idContinente']+0.20,valores,0.2,label="Periodo Jun- Nov Tipo: Méndez",color="#FDBD31")
        plt.xticks(x+1.1,continentes)
        plt.xlabel('Continentes')
        plt.ylabel('Toneladas exportadas')
        plt.title("Gráfica de  por temporada")
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)

        plt.legend()
        #plt.show()

        fig.savefig('static/imagenes/destinos_Temporada.png')

        tabla = tabla.to_html(classes='table table-striped', table_id="destinos")
        context = {'tabla': tabla}  # change

        return render(request, 'destinos.html', context)
    else:
        return redirect("/login/")

# Grafica de todos los paises paises por temporada
# ------------------------------------------------------------------------------------------------------------
def paisesTemporada(request):
    if request.user.is_authenticated:
        plt.close()
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")
        df = pd.DataFrame({"Empacadora":datos["Empacadora"], "Fecha":datos["Fecha expedición"], "Variedad":datos["Variedad"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"], "Pais":datos["País Destino"],})
        
        df.loc[(df.Unidad=="Kilogramos"),'Cantidad']=(df['Cantidad']/10000)
        df.loc[(df.Unidad=="Kilogramos"),'Unidad']="Toneladas"

    # Recibe el año seleccionado
        year = request.POST.get("year")
        pais = request.POST.get("pais")
        date1p1=year+'-06-01'
        date2p1=year+'-11-30'
        date1p2=year+'-12-01'
        yearAux = int(year)
        yearAux = yearAux+1
        year2 = str(yearAux)
        date2p2=year2+'-05-31'
        
        periodo1 =df.loc[df["Fecha"].between(date1p1, date2p1) & (df["Variedad"]=="Méndez")]
        periodo1 = pd.DataFrame(periodo1)
        periodo1 = periodo1.groupby(["Pais"], as_index=False)['Cantidad'].sum()
        periodo1 = pd.DataFrame(periodo1)


        periodo1 = periodo1[periodo1['Pais'] == pais] 
        periodo1 = pd.DataFrame(periodo1)
        

        periodo2 =df.loc[df["Fecha"].between(date1p2, date2p2) &(df["Variedad"]=="Hass")]
        periodo2 = pd.DataFrame(periodo2)
        periodo2 = periodo2.groupby(["Pais"], as_index=False)['Cantidad'].sum()
        periodo2 = pd.DataFrame(periodo2)


        periodo2 = periodo2[periodo2['Pais'] == pais] 
        periodo2 = pd.DataFrame(periodo2)
        


        pd.options.display.float_format = '{:,.2f}'.format
        tabla = pd.DataFrame({"Cantidad Méndez":periodo1['Cantidad'],"Cantidad Hass":periodo2['Cantidad']})
        tabla.set_index(['Cantidad Méndez','Cantidad Hass'], inplace = True)
        valores = periodo1['Cantidad']
        valores1 = periodo2['Cantidad']

        # Graficando valores
        valores = periodo1['Cantidad']
        valores1 = periodo2['Cantidad']

    # Graficando valores
        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')
        fig = plt.figure(figsize =(12,6))
        plt.barh('Hass',valores1,0.3,label="Periodo Dic- May Tipo: Hass",color="green")
        plt.barh('Méndez',valores,0.3,label="Periodo Jun- Nov Tipo: Méndez",color="#FDBD31")

        plt.ylabel('Tipo')
        plt.xlabel('Toneladas')
        plt.title("Gráfica de exportación del país "+pais)
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)
        plt.legend()
        #plt.show()

        fig.savefig('static/imagenes/paises_Temporada.png')
        tabla = tabla.to_html(classes='table table-striped', table_id="pais")
        context = {'tabla': tabla}  # change

        return render(request, 'paises.html', context)
    else:
        return redirect("/login/")


# PRONÓSTICOS
# ----------------------------------------------------------------------------------------------------------
#Continentes
def pronosticoPaises(request):
    if request.user.is_authenticated:
        plt.close()
        idContinente = request.POST.get("idContinente")
        destino = ""
        if (idContinente == "1"):
            destino = "Medio Oriente"
        else:
            if (idContinente == "2"):
                destino = "América"
            else:
                if (idContinente == "3"):
                    destino = "Europa"
                else:
                    destino = "Asia"
        tipoGrafica = request.POST.get("tipoGrafica")
        tipoPronostico = request.POST.get("tipoPronostico")

        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")

        datos = pd.DataFrame({'Fecha':datos["Fecha expedición"], "Variedad":datos["Variedad"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"],"idContinente":datos["Id Continente"]})
        
        datos.loc[(datos.Unidad=="Kilogramos"),'Cantidad']=(datos['Cantidad']/1000)
        datos.loc[(datos.Unidad=="Kilogramos"),'Unidad']="Toneladas"


        datos['Año'] = datos['Fecha'].dt.year
        datos = pd.DataFrame(datos)

        datos = datos.groupby('idContinente')
        datos = pd.DataFrame(datos.get_group(int(idContinente)))

        datos = datos.groupby("Año", as_index=False)["Cantidad"].sum()
        x = np.arange(len(datos.Año))
        datos["Periodo"] = x+1

        arregloPeriodo=[]
        arregloReal = []
        arregloLineal = []
        arregloRedes=[]
        arregloAños=[]
        longitudPeriodo = len(datos["Periodo"])
        log = longitudPeriodo
        logFuturo = log+6
        for log in range (log+6):
            arregloPeriodo.append(log + 1)
            if log < longitudPeriodo:
                arregloAños.append(datos.Año[log])
            else:
                arregloAños.append(arregloAños[-1]+1)

        datosP = pd.DataFrame({"Periodo":arregloPeriodo})    
        datos2 = pd.DataFrame({"Periodo":arregloPeriodo[longitudPeriodo:logFuturo]})

        #---------------------RED NEURONAL------------------------------
        
        print("Seleccionando las columnas")
        X_train = datos['Periodo']
        y_train = datos['Cantidad']  # Creando el Modelo
        print("Creando el modelo")
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
        
        print("Compilando el modelo")
        model.compile(optimizer=tf.keras.optimizers.Adam(1),
                    loss='mean_squared_error')

        print("Entrenando el modelo")
        epochs_hist = model.fit(X_train, y_train, epochs=15000)

        print("Evaluando el modelo entrenado")
        print("Keys:")
        print(epochs_hist.history.keys())

        #Regresión lineal

        linear_model = smf.ols(formula='Cantidad ~ Periodo', data=datos).fit()
        linear_model.params
        # Para realizar una predicción de cada fila en base a la variable 
        prediccionLineal = linear_model.predict(pd.DataFrame(datosP["Periodo"]))

        #LLENADO DE ARREGLOS
        for contador in range((len(datosP["Periodo"]))):
            # Realizar una predicción utilizando el modelo enTrenado
            Venta_C = contador + 1
            Venta_F = model.predict([Venta_C])  # red neuronal
            arregloRedes.append(float(Venta_F))
            print("Ventas de Prediccion: " + str(Venta_F))
            arregloLineal.append(prediccionLineal[contador])

        for cont in range(len(datos["Cantidad"])):
            arregloReal.append(datos.Cantidad[cont])
    
        #TABLAS
        pd.options.display.float_format = '{:,.2f}'.format
        tablaComparacion = pd.DataFrame({"Años":arregloAños[0:longitudPeriodo], 'Ventas Reales':arregloReal, "Regresión Lineal":arregloLineal[0:longitudPeriodo], "Redes Neuronales":arregloRedes[0:longitudPeriodo]})
        tablaComparacion.set_index('Años', inplace = True)
        tablaPronostico = pd.DataFrame({"Años":arregloAños[longitudPeriodo:logFuturo], "Regresión Lineal":arregloLineal[longitudPeriodo:logFuturo], "Redes Neuronales":arregloRedes[longitudPeriodo:logFuturo]})
        tablaPronostico.set_index('Años', inplace = True)
        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')
        plt.close()
        if tipoPronostico == "ventasReales":
            fig = plt.figure(figsize=(12, 6))
            plt.title('Pronóstico con comparación de ventas reales al Continente ' + destino)
            plt.ylabel('Toneladas de Aguacate Exportado')
            plt.xticks(x+1.25,arregloAños[0:longitudPeriodo])
            tabla = tablaComparacion.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos['Periodo']+0.00,arregloReal,0.20,label="Ventas Reales",  color="#FDBD31")
                plt.bar(datos['Periodo']+0.25, arregloLineal[0:longitudPeriodo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos['Periodo']+0.50, arregloRedes[0:longitudPeriodo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
            else: 
                plt.plot(datos['Periodo'],arregloReal,label="Ventas Reales",  color="#FDBD31")
                plt.plot(datos['Periodo'], arregloLineal[0:longitudPeriodo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos['Periodo'], arregloRedes[0:longitudPeriodo],label="Predicción: Redes Neuronales",color="#41A148")
        else:
            fig = plt.figure(figsize=(12, 10))
            plt.title('Pronóstico de exportaciones futuras al Continente ' + destino)
            plt.ylabel('Toneladas de Aguacate a exportar')
            plt.xticks(datos2['Periodo']+0.10,arregloAños[longitudPeriodo:logFuturo])
            tabla = tablaPronostico.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos2['Periodo']+0.00, arregloLineal[longitudPeriodo:logFuturo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos2['Periodo']+0.25, arregloRedes[longitudPeriodo:logFuturo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
                
            else:
                plt.plot(datos2['Periodo'], arregloLineal[longitudPeriodo:logFuturo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos2['Periodo'], arregloRedes[longitudPeriodo:logFuturo],label="Predicción: Redes Neuronales",color="#41A148")
            
        plt.xlabel('Años')
        plt.legend()
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)
        

        fig.savefig('static/imagenes/pronostico_destino.png')
        context = {'tabla': tabla}  # change

        return render(request, 'pronosticoPais.html',context)
    else:
        return redirect("/login/")

def pronosticoPaisesGeneral(request):
    if request.user.is_authenticated:
        plt.close()
        
        paisR = request.POST.get("pais")
        tipoGrafica = request.POST.get("tipoGrafica")
        tipoPronostico = request.POST.get("tipoPronostico")
        print(paisR)
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")

        datos = pd.DataFrame({'Fecha':datos["Fecha expedición"], "Variedad":datos["Variedad"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"],"Destino": datos["País Destino"]})

        datos.loc[(datos.Unidad=="Kilogramos"),'Cantidad']=(datos['Cantidad']/1000)
        datos.loc[(datos.Unidad=="Kilogramos"),'Unidad']="Toneladas"

        datos['Año'] = datos['Fecha'].dt.year
        datos = pd.DataFrame(datos)
        datos = datos.groupby('Destino')
        datos = pd.DataFrame(datos.get_group(paisR))
        datos = datos.groupby("Año", as_index=False)["Cantidad"].sum()
        x = np.arange(len(datos.Año))
        datos["Periodo"] = x+1

        arregloPeriodo=[]
        arregloReal = []
        arregloLineal = []
        arregloRedes=[]
        arregloAños=[]
        longitudPeriodo = len(datos["Periodo"])
        log = longitudPeriodo
        logFuturo = log+6
        for log in range (log+6):
            arregloPeriodo.append(log + 1)
            if log < longitudPeriodo:
                arregloAños.append(datos.Año[log])
            else:
                arregloAños.append(arregloAños[-1]+1)

        datosP = pd.DataFrame({"Periodo":arregloPeriodo})    
        datos2 = pd.DataFrame({"Periodo":arregloPeriodo[longitudPeriodo:logFuturo]})

        # Red Neuronal
        print("Seleccionando las columnas")
        X_train = datos['Periodo']
        y_train = datos['Cantidad']  # Creando el Modelo
        #messages.success(request,'cargando')
        print("Creando el modelo")
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

        print("Compilando el modelo")
        model.compile(optimizer=tf.keras.optimizers.Adam(1),
                        loss='mean_squared_error')
        print("Entrenando el modelo")
        epochs_hist = model.fit(X_train, y_train, epochs=15000)

        print("Evaluando el modelo entrenado")
        print("Keys:")
        print(epochs_hist.history.keys())    

        # Modelo Lineal
        linear_model = smf.ols(formula='Cantidad ~ Periodo', data=datos).fit()
        linear_model.params
        # Para realizar una predicción de cada fila en base a la variable de televisión
        prediccionLineal = linear_model.predict(pd.DataFrame(datosP["Periodo"]))

        for contador in range((len(datosP["Periodo"]))):
            # Realizar una predicción utilizando el modelo enTrenado
            Venta_C = contador + 1
            Venta_F = model.predict([Venta_C])  # red neuronal
            arregloRedes.append(float(Venta_F))
            print("Ventas de Prediccion: " + str(Venta_F))
            arregloLineal.append(prediccionLineal[contador])

        for cont in range(len(datos["Cantidad"])):
            arregloReal.append(datos.Cantidad[cont])

        pd.options.display.float_format = '{:,.2f}'.format
        tablaComparacion = pd.DataFrame({"Años":arregloAños[0:longitudPeriodo], 'Ventas Reales':arregloReal, "Regresión Lineal":arregloLineal[0:longitudPeriodo], "Redes Neuronales":arregloRedes[0:longitudPeriodo]})
        tablaComparacion.set_index('Años', inplace = True)
        tablaPronostico = pd.DataFrame({"Años":arregloAños[longitudPeriodo:logFuturo], "Regresión Lineal":arregloLineal[longitudPeriodo:logFuturo], "Redes Neuronales":arregloRedes[longitudPeriodo:logFuturo]})
        tablaPronostico.set_index('Años', inplace = True)
        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')
        plt.close()
        if tipoPronostico == "ventasReales":
            fig = plt.figure(figsize=(12, 6))
            plt.title('Pronóstico con comparación de ventas reales del país ' + paisR)
            plt.ylabel('Toneladas de Aguacate Exportado')
            plt.xticks(x+1.25,arregloAños[0:longitudPeriodo])
            tabla = tablaComparacion.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos['Periodo']+0.00,arregloReal,0.20,label="Ventas Reales",  color="#FDBD31")
                plt.bar(datos['Periodo']+0.25, arregloLineal[0:longitudPeriodo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos['Periodo']+0.50, arregloRedes[0:longitudPeriodo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
            else: 
                plt.plot(datos['Periodo'],arregloReal,label="Ventas Reales",  color="#FDBD31")
                plt.plot(datos['Periodo'], arregloLineal[0:longitudPeriodo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos['Periodo'], arregloRedes[0:longitudPeriodo],label="Predicción: Redes Neuronales",color="#41A148")
        else:
            fig = plt.figure(figsize=(12, 6))
            plt.title('Pronóstico de exportaciones futuras del país ' + paisR)
            plt.ylabel('Toneladas de Aguacate a exportar')
            plt.xticks(datos2['Periodo']+0.10,arregloAños[longitudPeriodo:logFuturo])
            tabla = tablaPronostico.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos2['Periodo']+0.00, arregloLineal[longitudPeriodo:logFuturo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos2['Periodo']+0.25, arregloRedes[longitudPeriodo:logFuturo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
                
            else:
                plt.plot(datos2['Periodo'], arregloLineal[longitudPeriodo:logFuturo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos2['Periodo'], arregloRedes[longitudPeriodo:logFuturo],label="Predicción: Redes Neuronales",color="#41A148")
            
        plt.xlabel('Años')
        plt.legend()
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)
        

        fig.savefig('static/imagenes/pronostico_pais.png')
        context = {'tabla': tabla}  # change
        return render(request, 'pronosticoGeneralPaises.html',context)
    else:
        return redirect("/login/")

def pronosticoEmpacadora(request):
    if request.user.is_authenticated:
        plt.close()
        empacadora = request.POST.get("empacadora")
        tipoGrafica = request.POST.get("tipoGrafica")
        tipoPronostico = request.POST.get("tipoPronostico")
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")

        datos = pd.DataFrame({"Empacadora":datos["Empacadora"], 'Fecha':datos["Fecha expedición"], "Variedad":datos["Variedad"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"]})

        datos.loc[(datos.Unidad=="Kilogramos"),'Cantidad']=(datos['Cantidad']/1000)
        datos.loc[(datos.Unidad=="Kilogramos"),'Unidad']="Toneladas"

        datos['Año'] = datos['Fecha'].dt.year
        datos = pd.DataFrame(datos)
        datos = datos[datos['Empacadora'] == empacadora]
        datos = datos.groupby("Año", as_index=False)["Cantidad"].sum()
        x = np.arange(len(datos.Año))
        datos["Periodo"] = x+1

        arregloPeriodo=[]
        arregloReal = []
        arregloLineal = []
        arregloRedes=[]
        arregloAños=[]
        longitudPeriodo = len(datos["Periodo"])
        log = longitudPeriodo
        logFuturo = log+6
        for log in range (log+6):
            arregloPeriodo.append(log + 1)
            if log < longitudPeriodo:
                arregloAños.append(datos.Año[log])
            else:
                arregloAños.append(arregloAños[-1]+1)

        datosP = pd.DataFrame({"Periodo":arregloPeriodo})    
        datos2 = pd.DataFrame({"Periodo":arregloPeriodo[longitudPeriodo:logFuturo]})
        # Red Neuronal
        print("Seleccionando las columnas")
        X_train = datos['Periodo']
        y_train = datos['Cantidad']  # Creando el Modelo
        print("Creando el modelo")
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

        print("Compilando el modelo")
        model.compile(optimizer=tf.keras.optimizers.Adam(1),
                        loss='mean_squared_error')
        print("Entrenando el modelo")
        epochs_hist = model.fit(X_train, y_train, epochs=15000)

        print("Evaluando el modelo entrenado")
        print("Keys:")
        print(epochs_hist.history.keys())

        # Modelo Lineal
        linear_model = smf.ols(formula='Cantidad ~ Periodo', data=datos).fit()
        linear_model.params
        # Para realizar una predicción de cada fila en base a la variable de televisión
        prediccionLineal = linear_model.predict(pd.DataFrame(datosP["Periodo"]))

        for contador in range((len(datosP["Periodo"]))):
            # Realizar una predicción utilizando el modelo enTrenado
            Venta_C = contador + 1
            Venta_F = model.predict([Venta_C])  # red neuronal
            arregloRedes.append(float(Venta_F))
            print("Ventas de Prediccion: " + str(Venta_F))
            arregloLineal.append(prediccionLineal[contador])

        for cont in range(len(datos["Cantidad"])):
            arregloReal.append(datos.Cantidad[cont])
        
        pd.options.display.float_format = '{:,.2f}'.format
        tablaComparacion = pd.DataFrame({"Años":arregloAños[0:longitudPeriodo], 'Ventas Reales':arregloReal, "Regresión Lineal":arregloLineal[0:longitudPeriodo], "Redes Neuronales":arregloRedes[0:longitudPeriodo]})
        tablaComparacion.set_index('Años', inplace = True)
        tablaPronostico = pd.DataFrame({"Años":arregloAños[longitudPeriodo:logFuturo], "Regresión Lineal":arregloLineal[longitudPeriodo:logFuturo], "Redes Neuronales":arregloRedes[longitudPeriodo:logFuturo]})
        tablaPronostico.set_index('Años', inplace = True)
        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')
        plt.close()
        if tipoPronostico == "ventasReales":
            fig = plt.figure(figsize=(12, 6))
            plt.title('Pronóstico con comparación de ventas reales de la empacadora ' + empacadora)
            plt.ylabel('Toneladas de Aguacate Exportado')
            plt.xticks(x+1.25,arregloAños[0:longitudPeriodo])
            tabla = tablaComparacion.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos['Periodo']+0.00,arregloReal,0.20,label="Ventas Reales",  color="#FDBD31")
                plt.bar(datos['Periodo']+0.25, arregloLineal[0:longitudPeriodo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos['Periodo']+0.50, arregloRedes[0:longitudPeriodo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
            else: 
                plt.plot(datos['Periodo'],arregloReal,label="Ventas Reales",  color="#FDBD31")
                plt.plot(datos['Periodo'], arregloLineal[0:longitudPeriodo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos['Periodo'], arregloRedes[0:longitudPeriodo],label="Predicción: Redes Neuronales",color="#41A148")
        else:
            fig = plt.figure(figsize=(12, 6))
            plt.title('Pronóstico de exportaciones futuras de la empacadora ' + empacadora)
            plt.ylabel('Toneladas de Aguacate a exportar')
            plt.xticks(datos2['Periodo']+0.10,arregloAños[longitudPeriodo:logFuturo])
            tabla = tablaPronostico.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos2['Periodo']+0.00, arregloLineal[longitudPeriodo:logFuturo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos2['Periodo']+0.25, arregloRedes[longitudPeriodo:logFuturo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
                
            else:
                plt.plot(datos2['Periodo'], arregloLineal[longitudPeriodo:logFuturo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos2['Periodo'], arregloRedes[longitudPeriodo:logFuturo],label="Predicción: Redes Neuronales",color="#41A148")
            
        plt.xlabel('Años')
        plt.legend()
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)
        

        fig.savefig('static/imagenes/pronostico_empacadora.png')
        context = {'tabla': tabla}  # change
        return render(request, 'pronosticoEmpacadora.html', context)
    else:
        return redirect("/login/")

def pronosticoGeneral(request):
    if request.user.is_authenticated:
        plt.close()
        tipoGrafica = request.POST.get("tipoGrafica")
        tipoPronostico = request.POST.get("tipoPronostico")
        datos = pd.read_csv("ProyectoApeajal/static/csv/certificados.csv")

        datos = pd.DataFrame({ 'Fecha':datos["Fecha expedición"], "Cantidad":datos["Cantidad"].str.replace(',', '').astype(float), "Unidad":datos["Uni. Medida"]})

        datos.loc[(datos.Unidad=="Kilogramos"),'Cantidad']=(datos['Cantidad']/1000)
        datos.loc[(datos.Unidad=="Kilogramos"),'Unidad']="Toneladas"

        datos['Año'] = datos['Fecha'].dt.year
        datos = pd.DataFrame(datos)
        datos = datos.groupby("Año", as_index=False)["Cantidad"].sum()
        x = np.arange(len(datos.Año))
        datos["Periodo"] = x+1

        arregloPeriodo=[]
        arregloReal = []
        arregloLineal = []
        arregloRedes=[]
        arregloAños=[]
        longitudPeriodo = len(datos["Periodo"])
        log = longitudPeriodo
        logFuturo = log+6
        for log in range (log+6):
            arregloPeriodo.append(log + 1)
            if log < longitudPeriodo:
                arregloAños.append(datos.Año[log])
            else:
                arregloAños.append(arregloAños[-1]+1)

        datosP = pd.DataFrame({"Periodo":arregloPeriodo})    
        datos2 = pd.DataFrame({"Periodo":arregloPeriodo[longitudPeriodo:logFuturo]})
        # Red Neuronal
        print("Seleccionando las columnas")
        X_train = datos['Periodo']
        y_train = datos['Cantidad']  # Creando el Modelo
        print("Creando el modelo")
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

        print("Compilando el modelo")
        model.compile(optimizer=tf.keras.optimizers.Adam(1),
                        loss='mean_squared_error')
        print("Entrenando el modelo")
        epochs_hist = model.fit(X_train, y_train, epochs=15000)

        print("Evaluando el modelo entrenado")
        print("Keys:")
        print(epochs_hist.history.keys())

        # Modelo Lineal
        linear_model = smf.ols(formula='Cantidad ~ Periodo', data=datos).fit()
        linear_model.params
        # Para realizar una predicción de cada fila en base a la variable de televisión
        prediccionLineal = linear_model.predict(pd.DataFrame(datosP["Periodo"]))

        for contador in range((len(datosP["Periodo"]))):
            # Realizar una predicción utilizando el modelo enTrenado
            Venta_C = contador + 1
            Venta_F = model.predict([Venta_C])  # red neuronal
            arregloRedes.append(float(Venta_F))
            print("Ventas de Prediccion: " + str(Venta_F))
            arregloLineal.append(prediccionLineal[contador])

        for cont in range(len(datos["Cantidad"])):
            arregloReal.append(datos.Cantidad[cont])
        
        pd.options.display.float_format = '{:,.2f}'.format
        tablaComparacion = pd.DataFrame({"Años":arregloAños[0:longitudPeriodo], 'Ventas Reales':arregloReal, "Regresión Lineal":arregloLineal[0:longitudPeriodo], "Redes Neuronales":arregloRedes[0:longitudPeriodo]})
        tablaComparacion.set_index('Años', inplace = True)
        tablaPronostico = pd.DataFrame({"Años":arregloAños[longitudPeriodo:logFuturo], "Regresión Lineal":arregloLineal[longitudPeriodo:logFuturo], "Redes Neuronales":arregloRedes[longitudPeriodo:logFuturo]})
        tablaPronostico.set_index('Años', inplace = True)
        im = image.imread('ProyectoApeajal/static/imagenes/marcadeagua.png')
        plt.close()
        if tipoPronostico == "ventasReales":
            fig = plt.figure(figsize=(12, 6))
            plt.title('Pronóstico con comparación de ventas reales por año')
            plt.ylabel('Toneladas de Aguacate Exportado')
            plt.xticks(x+1.25,arregloAños[0:longitudPeriodo])
            tabla = tablaComparacion.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos['Periodo']+0.00,arregloReal,0.20,label="Ventas Reales",  color="#FDBD31")
                plt.bar(datos['Periodo']+0.25, arregloLineal[0:longitudPeriodo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos['Periodo']+0.50, arregloRedes[0:longitudPeriodo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
            else: 
                plt.plot(datos['Periodo'],arregloReal,label="Ventas Reales",  color="#FDBD31")
                plt.plot(datos['Periodo'], arregloLineal[0:longitudPeriodo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos['Periodo'], arregloRedes[0:longitudPeriodo],label="Predicción: Redes Neuronales",color="#41A148")
        else:
            fig = plt.figure(figsize=(12, 6))
            plt.title('Pronóstico de exportaciones futuras por año')
            plt.ylabel('Toneladas de Aguacate a exportar')
            plt.xticks(datos2['Periodo']+0.10,arregloAños[longitudPeriodo:logFuturo])
            tabla = tablaPronostico.to_html(classes='table table-striped')
            if tipoGrafica == "Barras":
                plt.bar(datos2['Periodo']+0.00, arregloLineal[longitudPeriodo:logFuturo],0.20,label="Predicción: Regresión lineal", color="#C1CA31")
                plt.bar(datos2['Periodo']+0.25, arregloRedes[longitudPeriodo:logFuturo],0.20,label="Predicción: Redes Neuronales",color="#41A148")
                
            else:
                plt.plot(datos2['Periodo'], arregloLineal[longitudPeriodo:logFuturo],label="Predicción: Regresión lineal", color="#C1CA31")
                plt.plot(datos2['Periodo'], arregloRedes[longitudPeriodo:logFuturo],label="Predicción: Redes Neuronales",color="#41A148")
            
        plt.xlabel('Años')
        plt.legend()
        fig.figimage(im, 475, 200, zorder=3, alpha=.2)
        

        fig.savefig('static/imagenes/pronostico_General.png')
        context = {'tabla': tabla}  # change
        return render(request, 'pronosticoGeneral.html', context)
    else:
        return redirect("/login/")

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile =  request.FILES['myfile']
        name = str(myfile.name)
        print(name)
        
        if os.path.exists("ProyectoApeajal/static/csv/certificados.csv"):
            os.remove("ProyectoApeajal/static/csv/certificados.csv")
        fs = FileSystemStorage()
        filename = fs.save("certificados.csv",myfile)
        uploaded_file_url = fs.url(filename)
        print("Archivo correcto: "+ uploaded_file_url)
        messages.success(request,'Archivo cargado')
    else:
        print("Error")
        messages.success(request,'Ocurrio un error')
    return redirect("/login/index/")
    
