from django.db import models 
from django.contrib.auth.models import User

# Create your models here.

class Account(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    tipoUsuario = models.CharField(
        max_length=15,
        choices=[('administrador','administrador'),('empacadora','empacadora'),('estadistica','estadistica')]
        )
    
    def __str__(self):
        return self.user.username

