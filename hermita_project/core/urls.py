from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = "core"

urlpatterns = [
    path("", views.home, name="home"),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='core/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('properties/new/', views.property_create, name='property_create'),
    path('properties/', views.property_list, name='property_list'),
    path('properties/<int:property_id>/', views.property_detail, name='property_detail'),
    path('about/', views.about, name='about'), # New about page URL
    path('contact/', views.contact, name='contact'), # New contact page URL
]
