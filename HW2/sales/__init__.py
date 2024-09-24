# __init__.py
# Initialize the sales package

from .billing import process_billing
from .delivery import schedule_delivery
from .order import create_order


__version__ = "1.0"
