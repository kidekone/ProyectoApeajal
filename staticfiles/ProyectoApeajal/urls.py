"""ProyectoApeajal URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from ProyectoApeajal import views
from ProyectoApeajal.views import index
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/',views.login_request),
    path('login/index/',views.index),
    path('logout/',views.logout_request),
    path('paisesTemporada/',views.paisesTemporada),
    path('paises/',views.paises),
    path('pronosticoPais/',views.proPaises),
    path('pronosticoPaisSelecionado/',views.pronosticoPaises),
    path('pronosticoGeneralPaises/',views.proGeneralPaises),
    path('pronosticoGeneralPaisSeleccionado/',views.pronosticoPaisesGeneral),
    path('pronosticoEmpacadora/',views.proEmpacadora),
    path('pronosticoGeneral/',views.pronosticoGeneral),
    path('pronosticoEmpacadoraSeleccionada/',views.pronosticoEmpacadora),
    path('destinosTemporada/',views.destinosTemporada),
    path('destinos/',views.destinos),
    path('empacadorasGeneral/',views.empacadorasGeneral),
    path('empacadoraG/',views.empacadoraG),
    path('empacadorasIndividual/',views.empacadorasIndividual),
    path('empacadoraI/',views.empacadoraI),
    path('subirArchivo/',views.simple_upload),
]
