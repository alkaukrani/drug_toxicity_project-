
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from rdkit.Chem import MolFromSmiles # type: ignore



data = pd.read_csv('tox21 (3).csv')

#DATA PREPROCESSING



#focusing on three endpoints : NR-ER, SR-ARE,SR-MMP' as it had more 1s so DROPPING OTHER
data = data.drop(columns=['NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ATAD5', 'SR-HSE', 'SR-p53', 'mol_id', 'NR-AhR'])

#removing missing values 
data = data.dropna(subset=['NR-ER', 'SR-ARE', 'SR-MMP', 'smiles'])

print(data.columns)


# Generate Fingerprints
fingerprints=[]

valid_indices = []
for i, s in enumerate(data['smiles']):
    mol = MolFromSmiles(s)
    if mol is None:
        print(f"Invalid SMILES at index {i}: {s}")
    else:
        
        valid_indices.append(i)
        # Generate Morgan Fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        # Convert to numpy array
        fp_array = np.zeros((1, 2048))
        ConvertToNumpyArray(fp, fp_array[0])
        fingerprints.append(fp_array[0])

print(dir(AllChem)) 

# Keep only valid molecules
data = data.iloc[valid_indices].reset_index(drop=True)
       







#storing converted SMILES
x=np.array(fingerprints)

# Target Extraction (toxicity prediction)
y = data[['NR-ER', 'SR-ARE', 'SR-MMP']].values











