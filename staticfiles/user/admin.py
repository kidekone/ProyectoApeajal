from django.contrib import admin
from user.models import Account
from django.contrib.auth.admin import User
from django.contrib.auth.admin import UserAdmin

class AccountInLine(admin.StackedInline):
    model = Account
    can_delete = False
    verbose_name_plural = 'Accounts'

class CustomizedUserAdmin (UserAdmin):
    inlines = (AccountInLine,)
    
admin.site.unregister(User)
admin.site.register(User, CustomizedUserAdmin)
admin.site.register(Account)