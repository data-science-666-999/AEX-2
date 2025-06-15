from django.shortcuts import render, redirect, get_object_or_404 # Added get_object_or_404
from django.contrib.auth.models import User, Group
from django.contrib.auth import login
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.conf import settings # Added for MEDIA_URL

from .forms import UserRegistrationForm, PropertyForm # Assuming these forms
from .models import Property # Assuming Property model

def home(request):
    return render(request, "core/home.html")

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            role_name = form.cleaned_data['role']
            user = User.objects.create_user(username=username, email=email, password=password)
            group, created = Group.objects.get_or_create(name=role_name.capitalize())
            user.groups.add(group)
            login(request, user)
            messages.success(request, f"Registration successful! Welcome, {username}.")
            return redirect('core:home')
    else:
        form = UserRegistrationForm()
    return render(request, 'core/register.html', {'form': form})

@login_required
def dashboard(request):
    user_groups = [group.name for group in request.user.groups.all()]
    properties_list = [] # Renamed to avoid conflict if 'properties' is used elsewhere
    if "Landlord" in user_groups:
        properties_list = Property.objects.filter(landlord=request.user).order_by('-date_posted')

    return render(request, 'core/dashboard.html', {
        'user_groups': user_groups,
        'properties': properties_list, # Pass properties to the template
        'MEDIA_URL': settings.MEDIA_URL # Pass MEDIA_URL
    })

def is_landlord(user):
    return user.is_authenticated and user.groups.filter(name='Landlord').exists()

@user_passes_test(is_landlord)
def property_create(request):
    if request.method == 'POST':
        form = PropertyForm(request.POST, request.FILES)
        if form.is_valid():
            property_instance = form.save(commit=False)
            property_instance.landlord = request.user
            property_instance.save()
            messages.success(request, 'Property listed successfully!')
            return redirect('core:dashboard')
    else:
        form = PropertyForm()
    return render(request, 'core/property_create.html', {'form': form})

def property_list(request):
    # Fetch all available properties, ordered by date posted (newest first)
    available_properties = Property.objects.filter(is_available=True).order_by('-date_posted')

    # MEDIA_URL is already available via settings import at the top

    return render(request, 'core/property_list.html', {
        'properties': available_properties,
        'MEDIA_URL': settings.MEDIA_URL
    })

def property_detail(request, property_id):
    property_obj = get_object_or_404(Property, pk=property_id, is_available=True) # Fetch only available properties

    # MEDIA_URL is already available via settings import at the top

    return render(request, 'core/property_detail.html', {
        'property': property_obj,
        'MEDIA_URL': settings.MEDIA_URL
    })

def about(request):
    return render(request, 'core/about.html')

def contact(request):
    return render(request, 'core/contact.html')
