import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from rdkit.Chem import rdMolDescriptors # type: ignore
from rdkit.Chem import MolFromSmiles # type: ignore
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report



data = pd.read_csv('tox21 (3).csv')

#DATA PREPROCESSING



#focusing on three endpoints : NR-ER, SR-ARE,SR-MMP' as it had more 1s so DROPPING OTHER
data = data.drop(columns=['NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ATAD5', 'SR-HSE', 'SR-p53', 'mol_id', 'NR-AhR'])

#removing missing values 
data = data.dropna(subset=['NR-ER', 'SR-ARE', 'SR-MMP', 'smiles'])




# Generate Fingerprints
fingerprints=[]

valid_indices = []
for i, s in enumerate(data['smiles']):
    mol = MolFromSmiles(s)
    if mol is None:
        print(f"Invalid SMILES at index {i}: {s}")
    else:
        
        #valid smiles
        valid_indices.append(i)
        # Generate Morgan Fingerprints
        morgan_generator = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
        # Initialize the Morgan generator for generating fingerprints


     

        fp = morgan_generator.GetFingerprint(mol)
        # Convert to numpy array
        fp_array = np.zeros((1, 2048))
        ConvertToNumpyArray(fp, fp_array[0]) #refers to the first row of the placeholder fp_array, where the fingerprint will be stored
        fingerprints.append(fp_array[0])





# Keep only valid molecules
data = data.iloc[valid_indices].reset_index(drop=True)

#storing converted SMILES
x=np.array(fingerprints)

# Target Extraction (toxicity prediction)
y = data[['NR-ER', 'SR-ARE', 'SR-MMP']].values

gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# Train the model

# Train the model
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Train the model

gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))