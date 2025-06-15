from django.db import models
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _

class Property(models.Model):
    landlord = models.ForeignKey(User, on_delete=models.CASCADE, related_name='properties', verbose_name=_("Landlord"))
    title = models.CharField(_("Title"), max_length=200)
    description = models.TextField(_("Description"))
    address_line1 = models.CharField(_("Address Line 1"), max_length=255)
    city = models.CharField(_("City"), max_length=100)
    county = models.CharField(_("County/Region"), max_length=100) # Relevant for Kenya
    rent_amount = models.DecimalField(_("Rent Amount"), max_digits=10, decimal_places=2)
    bedrooms = models.PositiveIntegerField(_("Bedrooms"))
    bathrooms = models.PositiveIntegerField(_("Bathrooms"))
    square_footage = models.PositiveIntegerField(_("Square Footage"), null=True, blank=True)
    main_image = models.ImageField(_("Main Image"), upload_to='properties/%Y/%m/%d/', null=True, blank=True)
    is_available = models.BooleanField(_("Is Available"), default=True)
    date_posted = models.DateTimeField(_("Date Posted"), auto_now_add=True)
    last_updated = models.DateTimeField(_("Last Updated"), auto_now=True)

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = _("Property")
        verbose_name_plural = _("Properties")
        ordering = ['-date_posted']
