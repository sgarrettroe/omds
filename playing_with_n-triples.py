import rdflib
import pprint
g = rdflib.Graph()
g.parse("omds.ttl")
print(g.serialize(format="turtle"))
print(g.serialize(format="ntriples"))
with open("omds-units.nt", 'w') as file:
    file.write(g.serialize(format="ntriples"))

print('done')