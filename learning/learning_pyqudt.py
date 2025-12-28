from qudt.quantity import Quantity
from qudt.units.concentration import ConcentrationUnit
from qudt.units.temperature import TemperatureUnit
from learning_pyqudt2 import TimeUnit

obs = Quantity(0.1, ConcentrationUnit.MICROMOLAR)
print(f'{obs} = {obs.convert_to(ConcentrationUnit.NANOMOLAR)}')

temp = Quantity(20, TemperatureUnit.CELSIUS)
print(f'{temp} = {temp.convert_to(TemperatureUnit.KELVIN)}')

temp = Quantity(42, TimeUnit.SEC)
print(f'{temp}')
#print(f'{temp} = {temp.convert_to(TemperatureUnit.KELVIN)}')

