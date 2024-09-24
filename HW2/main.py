# main.py
import pricing


net_price = pricing.get_net_price(
    price=100,
    tax_rate=0.01
)

print(net_price)

import pricing as selling_price

net_price = selling_price.get_net_price(
    price=100,
    tax_rate=0.01
)
print(net_price)


from pricing import get_net_price

net_price = get_net_price(price=100, tax_rate=0.01)
print(net_price)

from pricing import get_net_price as calculate_net_price

net_price = calculate_net_price(
    price=100,
    tax_rate=0.1,
    discount=0.05
)
print(net_price)

from pricing import *
from Product import *

tax = get_tax(100)
print(tax)

import sys

for path in sys.path:
    print(path)
import sys

# Add the directory where recruitment.py is located
sys.path.append('asset/')
import os
asset_path = os.path.join(r'c:\Users\manoj\OneDrive\Desktop\Pattern_Recognition_and_NN\HW2', 'asset')

# Add the absolute path to 'sys.path'
sys.path.append(asset_path)

# Now import recruitment from the 'asset' folder
# #import recruitment

# # Use the functions from recruitment
# recruitment.hire()
# recruitment.fire()
# recruitment.promote()

import billing
