import os
from mp_api.client import MPRester
from pymatgen.core import Composition, Element

# Setup
API_KEY = "dnq07hvQEKLgvwySSmAi4QGzItrjZKpg"
mpr = MPRester(API_KEY)
os.makedirs('oxide_cifs', exist_ok=True)

# Search for binary and ternary metal oxides
docs = mpr.summary.search(
    num_elements=(2, 3),  # Binary and ternary only
    elements=["O"],       # Must contain oxygen
    fields=["material_id", "formula_pretty", "structure"]
)

# Filter for metal oxides and download
downloaded = 0
for doc in docs:
    try:
        comp = Composition(doc.formula_pretty)
        elements = comp.elements
        
        # Check if it's a metal oxide (has metals + oxygen)
        has_metal = any(el.is_metal for el in elements if el != Element("O"))
        
        if has_metal:
            doc.structure.to(filename=f'oxide_cifs/{doc.material_id}.cif')
            downloaded += 1
    except:
        continue

print(f"Downloaded {downloaded} binary/ternary metal oxide CIF files")