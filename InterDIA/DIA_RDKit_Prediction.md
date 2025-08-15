```python
import pandas as pd
```


```python
# 1.1 Load training dataset
training_set = pd.read_csv("DataSet/DIA_trainingset_RDKit_descriptors.csv")
training_set
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMILES</th>
      <th>BalabanJ</th>
      <th>BertzCT</th>
      <th>Chi0</th>
      <th>Chi0n</th>
      <th>Chi0v</th>
      <th>Chi1</th>
      <th>Chi1n</th>
      <th>Chi1v</th>
      <th>...</th>
      <th>fr_sulfide</th>
      <th>fr_sulfonamd</th>
      <th>fr_sulfone</th>
      <th>fr_term_acetylene</th>
      <th>fr_tetrazole</th>
      <th>fr_thiazole</th>
      <th>fr_thiocyan</th>
      <th>fr_thiophene</th>
      <th>fr_unbrch_alkane</th>
      <th>fr_urea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>COC(=O)N(C)c1c(N)nc(nc1N)c2nn(Cc3ccccc3F)c4ncc...</td>
      <td>1.821</td>
      <td>1266.407</td>
      <td>22.121</td>
      <td>16.781</td>
      <td>16.781</td>
      <td>14.901</td>
      <td>9.203</td>
      <td>9.203</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>C[C@H](N(O)C(=O)N)c1cc2ccccc2s1</td>
      <td>2.363</td>
      <td>490.434</td>
      <td>11.707</td>
      <td>8.752</td>
      <td>9.569</td>
      <td>7.592</td>
      <td>4.854</td>
      <td>5.670</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>C[N+](C)(C)CC(=O)[O-]</td>
      <td>3.551</td>
      <td>93.092</td>
      <td>6.784</td>
      <td>5.471</td>
      <td>5.471</td>
      <td>3.417</td>
      <td>2.420</td>
      <td>2.420</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CC(C)n1c(\C=C\[C@H](O)C[C@H](O)CC(=O)O)c(c2ccc...</td>
      <td>2.076</td>
      <td>1053.003</td>
      <td>21.836</td>
      <td>16.995</td>
      <td>16.995</td>
      <td>14.274</td>
      <td>9.926</td>
      <td>9.926</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>C\C(=C(\C#N)/C(=O)Nc1ccc(cc1)C(F)(F)F)\O</td>
      <td>2.888</td>
      <td>549.823</td>
      <td>14.629</td>
      <td>9.746</td>
      <td>9.746</td>
      <td>8.752</td>
      <td>5.040</td>
      <td>5.040</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0</td>
      <td>C(C1=NCCN1)c2cccc3ccccc23</td>
      <td>2.022</td>
      <td>537.932</td>
      <td>10.795</td>
      <td>9.110</td>
      <td>9.110</td>
      <td>7.933</td>
      <td>5.672</td>
      <td>5.672</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>473</th>
      <td>0</td>
      <td>C[N@+]1(CC2CC2)CC[C@]34[C@H]5Oc6c(O)ccc(C[C@@H...</td>
      <td>1.602</td>
      <td>848.658</td>
      <td>17.897</td>
      <td>15.202</td>
      <td>15.202</td>
      <td>12.389</td>
      <td>10.003</td>
      <td>10.003</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>474</th>
      <td>1</td>
      <td>CO\N=C(/C(=O)N[C@H]1[C@H]2SCC(=C(N2C1=O)C(=O)O...</td>
      <td>1.766</td>
      <td>910.031</td>
      <td>21.129</td>
      <td>14.986</td>
      <td>15.802</td>
      <td>13.845</td>
      <td>8.129</td>
      <td>9.178</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>475</th>
      <td>0</td>
      <td>Clc1ccc(CO\N=C(\Cn2ccnc2)/c3ccc(Cl)cc3Cl)c(Cl)c1</td>
      <td>1.831</td>
      <td>926.191</td>
      <td>18.518</td>
      <td>13.372</td>
      <td>16.396</td>
      <td>12.525</td>
      <td>7.566</td>
      <td>9.078</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>476</th>
      <td>1</td>
      <td>CCC[C@@]1(CCc2ccccc2)CC(=C([C@H](CC)c3cccc(c3)...</td>
      <td>1.617</td>
      <td>1565.385</td>
      <td>30.545</td>
      <td>23.226</td>
      <td>24.043</td>
      <td>19.871</td>
      <td>13.658</td>
      <td>15.098</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>477 rows × 198 columns</p>
</div>




```python
Xtrain = training_set.iloc[:, 2:]
Ytrain = training_set.iloc[:, 0]

# Display label distribution
print(Ytrain.value_counts())
```

    Label
    0    359
    1    118
    Name: count, dtype: int64
    


```python
# 1.2 Load test dataset
test_set = pd.read_csv("DataSet/DIA_testset_RDKit_descriptors.csv")
test_set
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMILES</th>
      <th>BalabanJ</th>
      <th>BertzCT</th>
      <th>Chi0</th>
      <th>Chi0n</th>
      <th>Chi0v</th>
      <th>Chi1</th>
      <th>Chi1n</th>
      <th>Chi1v</th>
      <th>...</th>
      <th>fr_sulfide</th>
      <th>fr_sulfonamd</th>
      <th>fr_sulfone</th>
      <th>fr_term_acetylene</th>
      <th>fr_tetrazole</th>
      <th>fr_thiazole</th>
      <th>fr_thiocyan</th>
      <th>fr_thiophene</th>
      <th>fr_unbrch_alkane</th>
      <th>fr_urea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>C[C@H](\C=C\[C@H](O)C1CC1)[C@@H]2CC[C@@H]3\C(=...</td>
      <td>1.484</td>
      <td>743.207</td>
      <td>21.466</td>
      <td>18.764</td>
      <td>18.764</td>
      <td>14.292</td>
      <td>12.106</td>
      <td>12.106</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>OCCN1CCN(CCCN2c3ccccc3Sc4ccc(cc24)C(F)(F)F)CC1</td>
      <td>1.472</td>
      <td>868.947</td>
      <td>21.140</td>
      <td>16.736</td>
      <td>17.553</td>
      <td>14.453</td>
      <td>10.268</td>
      <td>11.084</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>C[C@@H]1O[C@H](C[C@H](O)[C@@H]1O)O[C@@H]2[C@H]...</td>
      <td>0.837</td>
      <td>1409.004</td>
      <td>39.189</td>
      <td>32.904</td>
      <td>32.904</td>
      <td>26.011</td>
      <td>20.941</td>
      <td>20.941</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>NC(=O)Cc1cccc(C(=O)c2ccccc2)c1N</td>
      <td>2.406</td>
      <td>621.298</td>
      <td>13.828</td>
      <td>10.297</td>
      <td>10.297</td>
      <td>9.092</td>
      <td>5.847</td>
      <td>5.847</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>COc1cc2c(CCN[C@]23CS[C@@H]4[C@@H]5[C@@H]6N(C)[...</td>
      <td>1.320</td>
      <td>2127.996</td>
      <td>37.955</td>
      <td>30.849</td>
      <td>31.666</td>
      <td>25.910</td>
      <td>18.066</td>
      <td>19.115</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>0</td>
      <td>CCN1CCN(C(=O)N[C@H](C(=O)N[C@@H]2[C@@H]3SC(C)(...</td>
      <td>1.508</td>
      <td>1127.109</td>
      <td>26.361</td>
      <td>19.925</td>
      <td>20.742</td>
      <td>16.973</td>
      <td>11.450</td>
      <td>12.330</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>116</th>
      <td>0</td>
      <td>CC1=C(C=C(C#N)C(=O)N1)c2ccncc2</td>
      <td>2.678</td>
      <td>608.396</td>
      <td>11.544</td>
      <td>8.689</td>
      <td>8.689</td>
      <td>7.720</td>
      <td>4.765</td>
      <td>4.765</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>0</td>
      <td>CCCN(CCc1cccs1)[C@@H]2CCc3c(O)cccc3C2</td>
      <td>1.670</td>
      <td>593.488</td>
      <td>15.364</td>
      <td>13.294</td>
      <td>14.110</td>
      <td>10.775</td>
      <td>8.338</td>
      <td>9.217</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>0</td>
      <td>COCCOC(=O)C1=C(C)NC(=C([C@@H]1c2cccc(c2)[N+](=...</td>
      <td>2.603</td>
      <td>902.371</td>
      <td>22.422</td>
      <td>17.683</td>
      <td>17.683</td>
      <td>14.167</td>
      <td>9.469</td>
      <td>9.469</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>119</th>
      <td>0</td>
      <td>CCCCCCCCCCNCC=C</td>
      <td>2.814</td>
      <td>109.803</td>
      <td>10.485</td>
      <td>9.856</td>
      <td>9.856</td>
      <td>6.914</td>
      <td>6.231</td>
      <td>6.231</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 198 columns</p>
</div>




```python
Xtest = test_set.iloc[:, 2:]
Ytest = test_set.iloc[:, 0]
```


```python
# Display label distribution
print(Ytest.value_counts())
```

    Label
    0    90
    1    30
    Name: count, dtype: int64
    


```python
# 2. Feature preprocessing pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import joblib

def preprocess_features(X_train, X_test):
    """
    Preprocess features: standardization, variance threshold, correlation filtering
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        X_train_processed, X_test_processed (pd.DataFrame)
    """
    # 2.1. Check missing values
    print("\nMissing values:")
    print("Training set:", X_train.isnull().sum().sum())
    print("Test set:", X_test.isnull().sum().sum())
    
    # 2.2. Standardization
    scaler = StandardScaler()
    X_train_std = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_std = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for reproducibility
    
    # 2.3. Remove zero-variance features
    selector = VarianceThreshold()
    selector.fit(X_train_std)
    keep_vars = X_train.columns[selector.variances_ != 0].tolist()
    
    X_train_var = X_train_std[keep_vars]
    X_test_var = X_test_std[keep_vars]
    
    # 2.4. Remove highly correlated features
    corr_matrix = X_train_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_features = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    X_train_processed = X_train_var.drop(columns=drop_features)
    X_test_processed = X_test_var.drop(columns=drop_features)
    
    print(f"\nFeatures reduced from {X_train.shape[1]} to {X_train_processed.shape[1]}")
    
    # Save processed features
    X_train_processed.to_csv("X_train_processed2.csv", index=False)
    X_test_processed.to_csv("X_test_processed2.csv", index=False)
    
    return X_train_processed, X_test_processed

```


```python
# Preprocess features
X_train_processed, X_test_processed = preprocess_features(Xtrain, Xtest)
```

    
    Missing values:
    Training set: 0
    Test set: 0
    
    Features reduced from 196 to 140
    


```python
X_train_processed
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BalabanJ</th>
      <th>BertzCT</th>
      <th>Chi0</th>
      <th>EState_VSA1</th>
      <th>EState_VSA10</th>
      <th>EState_VSA11</th>
      <th>EState_VSA2</th>
      <th>EState_VSA3</th>
      <th>EState_VSA4</th>
      <th>EState_VSA5</th>
      <th>...</th>
      <th>fr_quatN</th>
      <th>fr_sulfide</th>
      <th>fr_sulfonamd</th>
      <th>fr_sulfone</th>
      <th>fr_term_acetylene</th>
      <th>fr_tetrazole</th>
      <th>fr_thiazole</th>
      <th>fr_thiophene</th>
      <th>fr_unbrch_alkane</th>
      <th>fr_urea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.452875</td>
      <td>1.344465</td>
      <td>0.551264</td>
      <td>-0.525136</td>
      <td>-0.384981</td>
      <td>-0.181361</td>
      <td>1.395076</td>
      <td>0.554831</td>
      <td>-0.812459</td>
      <td>0.507059</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.310322</td>
      <td>-0.632257</td>
      <td>-0.887194</td>
      <td>-0.263887</td>
      <td>-0.305248</td>
      <td>-0.181361</td>
      <td>-1.140124</td>
      <td>-0.755872</td>
      <td>-0.247407</td>
      <td>-0.915263</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>5.551415</td>
      <td>-0.238492</td>
      <td>5.974304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.983159</td>
      <td>-1.644450</td>
      <td>-1.567194</td>
      <td>-0.530553</td>
      <td>-0.315105</td>
      <td>-0.181361</td>
      <td>-0.672837</td>
      <td>-0.799998</td>
      <td>-1.087602</td>
      <td>-0.915263</td>
      <td>...</td>
      <td>6.833740</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.093806</td>
      <td>0.800838</td>
      <td>0.511898</td>
      <td>0.283293</td>
      <td>0.611727</td>
      <td>-0.181361</td>
      <td>0.164995</td>
      <td>-1.141063</td>
      <td>0.469142</td>
      <td>-0.061968</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.049581</td>
      <td>-0.480968</td>
      <td>-0.483586</td>
      <td>0.474686</td>
      <td>0.471975</td>
      <td>-0.181361</td>
      <td>-0.734095</td>
      <td>-1.141063</td>
      <td>0.663707</td>
      <td>-0.488440</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>472</th>
      <td>-0.169844</td>
      <td>-0.511260</td>
      <td>-1.013166</td>
      <td>-0.791321</td>
      <td>-1.281364</td>
      <td>-0.181361</td>
      <td>-1.140124</td>
      <td>-1.141063</td>
      <td>0.335614</td>
      <td>0.233622</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>473</th>
      <td>-0.761251</td>
      <td>0.280287</td>
      <td>-0.032186</td>
      <td>-0.043399</td>
      <td>0.183298</td>
      <td>-0.181361</td>
      <td>0.114661</td>
      <td>0.273335</td>
      <td>1.577292</td>
      <td>-0.012105</td>
      <td>...</td>
      <td>6.833740</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>474</th>
      <td>-0.530321</td>
      <td>0.436629</td>
      <td>0.414242</td>
      <td>0.750526</td>
      <td>1.088659</td>
      <td>-0.181361</td>
      <td>1.366018</td>
      <td>-1.141063</td>
      <td>-0.812459</td>
      <td>1.279054</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>3.862468</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>475</th>
      <td>-0.438794</td>
      <td>0.477795</td>
      <td>0.053591</td>
      <td>-0.791321</td>
      <td>-1.281364</td>
      <td>-0.181361</td>
      <td>-0.668410</td>
      <td>1.319889</td>
      <td>-0.462804</td>
      <td>-0.915263</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
    <tr>
      <th>476</th>
      <td>-0.740130</td>
      <td>2.106085</td>
      <td>1.714848</td>
      <td>1.142442</td>
      <td>1.825381</td>
      <td>-0.181361</td>
      <td>0.533398</td>
      <td>2.169167</td>
      <td>-0.074011</td>
      <td>0.364644</td>
      <td>...</td>
      <td>-0.146333</td>
      <td>-0.222670</td>
      <td>-0.204792</td>
      <td>-0.122039</td>
      <td>-0.079556</td>
      <td>-0.09196</td>
      <td>-0.168934</td>
      <td>-0.155535</td>
      <td>-0.238492</td>
      <td>-0.167384</td>
    </tr>
  </tbody>
</table>
<p>477 rows × 140 columns</p>
</div>




```python
"""
Step 3: Hyperparameter Optimization and Feature Selection for Balanced Random Forest

This script performs hyperparameter optimization for Balanced Random Forest Classifier 
on preprocessed features, followed by feature selection based on the optimized model.

Process:
1. Load preprocessed features (X_train_processed) and labels (Ytrain)
2. Optimize hyperparameters using Hyperopt with MCC as objective
3. Train model with best parameters for subsequent feature selection
"""
```




    '\nStep 3: Hyperparameter Optimization and Feature Selection for Balanced Random Forest\n\nThis script performs hyperparameter optimization for Balanced Random Forest Classifier \non preprocessed features, followed by feature selection based on the optimized model.\n\nProcess:\n1. Load preprocessed features (X_train_processed) and labels (Ytrain)\n2. Optimize hyperparameters using Hyperopt with MCC as objective\n3. Train model with best parameters for subsequent feature selection\n'




```python

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, matthews_corrcoef
from imblearn.ensemble import BalancedRandomForestClassifier
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import cross_validate, cross_val_score

def get_model_performance(X, y, model_params, random_state=1):
   """Get CV and OOF performance metrics for a model"""
   model = BalancedRandomForestClassifier(**model_params)
   cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
   cv_metrics = []
   
   # Get CV metrics
   for train_idx, val_idx in cv.split(X):
       X_train_fold = X.iloc[train_idx]
       X_val_fold = X.iloc[val_idx]
       y_train_fold = y.iloc[train_idx] 
       y_val_fold = y.iloc[val_idx]
       
       model.fit(X_train_fold, y_train_fold)
       y_pred = model.predict(X_val_fold)
       y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
       
       tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred).ravel()
       metrics = {
           'AUC': roc_auc_score(y_val_fold, y_pred_proba),
           'ACC': accuracy_score(y_val_fold, y_pred),
           'SEN': tp / (tp + fn),
           'SPE': tn / (tn + fp),
           'MCC': matthews_corrcoef(y_val_fold, y_pred)
       }
       cv_metrics.append(metrics)
   
   cv_mean = {m: np.mean([fold[m] for fold in cv_metrics]) for m in cv_metrics[0].keys()}
   
   # Get OOF predictions
   y_pred_all = np.zeros_like(y)
   y_pred_proba_all = np.zeros_like(y, dtype=float)
   
   for train_idx, val_idx in cv.split(X):
       X_train_fold = X.iloc[train_idx]
       X_val_fold = X.iloc[val_idx]
       y_train_fold = y.iloc[train_idx]
       
       model.fit(X_train_fold, y_train_fold)
       y_pred_all[val_idx] = model.predict(X_val_fold)
       y_pred_proba_all[val_idx] = model.predict_proba(X_val_fold)[:, 1]
   
   # Calculate OOF metrics
   tn, fp, fn, tp = confusion_matrix(y, y_pred_all).ravel()
   oof_metrics = {
       'AUC': roc_auc_score(y, y_pred_proba_all),
       'ACC': accuracy_score(y, y_pred_all),
       'SEN': tp / (tp + fn),
       'SPE': tn / (tn + fp),
       'MCC': matthews_corrcoef(y, y_pred_all)
   }
   
   # Create results table
   metrics_order = ['AUC', 'ACC', 'SEN', 'SPE', 'MCC']
   results = pd.DataFrame({
       'Metric': metrics_order,
       'CV Mean': [cv_mean[m] for m in metrics_order],
       'Out-of-fold': [oof_metrics[m] for m in metrics_order]
   })
   
   return results

def optimize_brf(X, y, random_state=1):
    def objective(params):
        model = BalancedRandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            criterion=params["criterion"],
            max_depth=int(params["max_depth"]),
            max_features=params["max_features"],
            sampling_strategy='auto',
            replacement=False,
            random_state=random_state,
            n_jobs=-1
        )
        cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring="matthews_corrcoef",
            n_jobs=-1,
            error_score="raise"
        )
        return -np.mean(scores["test_score"])

    max_features_choices = ["sqrt", "log2"] + list(range(10, 70))
    param_space = {
        'n_estimators': hp.quniform("n_estimators", 40, 300, 1),
        'criterion': hp.choice("criterion", ["gini", "entropy"]),
        'max_depth': hp.quniform('max_depth', 1, 25, 1),
        'max_features': hp.choice("max_features", max_features_choices)
    }
    
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
        early_stop_fn=no_progress_loss(400),
        verbose=True
    )
    
    # Convert parameters to correct types
    best_params = {
        "n_estimators": int(best_params["n_estimators"]),
        "criterion": ["gini", "entropy"][best_params["criterion"]],
        "max_depth": int(best_params["max_depth"]),
        "max_features": max_features_choices[best_params["max_features"]],
        "bootstrap": True,
        "sampling_strategy": 'auto',
        "replacement": False,
        "random_state": random_state,
        "verbose": False,
        "n_jobs": -1
    }
    
    return best_params
```


```python
# """
# Run optimization and evaluation
best_params = optimize_brf(X_train_processed, Ytrain)
results = get_model_performance(X_train_processed, Ytrain, best_params)
print("\nBest Parameters:", best_params)
print("\nModel Performance:")
print(results.to_string(index=False, float_format='%.4f'))
# """
```




    '\n# Run optimization and evaluation\nbest_params = optimize_brf(X_train_processed, Ytrain)\nresults = get_model_performance(X_train_processed, Ytrain, best_params)\nprint("\nBest Parameters:", best_params)\nprint("\nModel Performance:")\nprint(results.to_string(index=False, float_format=\'%.4f\'))\n'




```python
# Define best parameters for feature processing
best_params = {
   'n_estimators': 255,
   'criterion': 'gini',
   'max_depth': 16,
   'max_features': 30,
   'bootstrap': True,
   'sampling_strategy': 'auto',
   'replacement': False,
   'random_state': 1,
   'verbose': False,
   'n_jobs': -1
}
# Get model performance
results = get_model_performance(X_train_processed, Ytrain, best_params)
print("\nModel Performance:")
print(results.to_string(index=False, float_format='%.4f'))
```

    
    Model Performance:
    Metric  CV Mean  Out-of-fold
       AUC   0.8545       0.8617
       ACC   0.7820       0.7820
       SEN   0.7740       0.7881
       SPE   0.7806       0.7799
       MCC   0.5022       0.5105
    


```python
# best model from preprocessing
best_model_preprocessing = BalancedRandomForestClassifier(
   n_estimators=255,
   criterion='gini',
   max_depth=16,
   max_features=30,
   bootstrap=True,
   sampling_strategy='auto',
   replacement=False,
   random_state=1,
   verbose=False,  
   n_jobs=-1
)
best_model_preprocessing.fit(X_train_processed,Ytrain)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>BalancedRandomForestClassifier(bootstrap=True, max_depth=16, max_features=30,
                               n_estimators=255, n_jobs=-1, random_state=1,
                               replacement=False, sampling_strategy=&#x27;auto&#x27;,
                               verbose=False)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>BalancedRandomForestClassifier</div></div><div><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>BalancedRandomForestClassifier(bootstrap=True, max_depth=16, max_features=30,
                               n_estimators=255, n_jobs=-1, random_state=1,
                               replacement=False, sampling_strategy=&#x27;auto&#x27;,
                               verbose=False)</pre></div> </div></div></div></div>




```python
"""
Step 4: Feature Selection Methods

This script implements four different feature selection methods:
1. Mutual Information (MI)
2. Embedded tree-based feature selection (ETB)
3. Recursive Feature Elimination with Cross-Validation (RFECV)
4. Genetic Algorithm (GA)

Note: Only GA results will be demonstrated; other methods are provided as code templates.
"""
```




    '\nStep 4: Feature Selection Methods\n\nThis script implements four different feature selection methods:\n1. Mutual Information (MI)\n2. Embedded tree-based feature selection (ETB)\n3. Recursive Feature Elimination with Cross-Validation (RFECV)\n4. Genetic Algorithm (GA)\n\nNote: Only GA results will be demonstrated; other methods are provided as code templates.\n'




```python
# 4.1 Mutual Information Selection

from sklearn.feature_selection import mutual_info_classif as MIC, SelectKBest

def mutual_information_selection(X, y, random_state=0):
    """
    Mutual Information based feature selection
    """
    np.random.seed(random_state)
    
    # Calculate mutual information scores
    mi_scores = MIC(X, y)
    
    # Get number of features with MI > 0
    k = mi_scores.shape[0] - sum(mi_scores <= 0)
    
    # Select top k features
    selector = SelectKBest(MIC, k=k)
    selector.fit(X, y)
    
    # Get selected feature names and scores
    feature_scores = pd.Series(mi_scores, index=X.columns)
    selected_features = feature_scores[feature_scores > 0].index.tolist()
    
    # Sort and return scores
    feature_scores_sorted = feature_scores.sort_values(ascending=False)
    
    return selected_features, feature_scores_sorted, selector

# Example usage:
"""
selected_features_mi, mi_scores, mi_selector = mutual_information_selection(
    X_train_processed, Ytrain)
print(f"Number of selected features: {len(selected_features_mi)}")
print("\nTop 10 features by MI score:")
print(mi_scores.head(10))

# Get transformed dataset with selected features
X_train_mi = X_train_processed[selected_features_mi]

# Optimize hyperparameters for MI-selected features
best_params_mi = optimize_brf(X_train_mi, Ytrain)
print("\nBest parameters for MI-selected features:", best_params_mi)

# Evaluate performance with MI-selected features
results_mi = get_model_performance(X_train_mi, Ytrain, best_params_mi)
print("\nModel Performance with MI-selected features:")
print(results_mi.to_string(index=False, float_format='%.4f'))
"""
```




    '\nselected_features_mi, mi_scores, mi_selector = mutual_information_selection(\n    X_train_processed, Ytrain)\nprint(f"Number of selected features: {len(selected_features_mi)}")\nprint("\nTop 10 features by MI score:")\nprint(mi_scores.head(10))\n\n# Get transformed dataset with selected features\nX_train_mi = X_train_processed[selected_features_mi]\n\n# Optimize hyperparameters for MI-selected features\nbest_params_mi = optimize_brf(X_train_mi, Ytrain)\nprint("\nBest parameters for MI-selected features:", best_params_mi)\n\n# Evaluate performance with MI-selected features\nresults_mi = get_model_performance(X_train_mi, Ytrain, best_params_mi)\nprint("\nModel Performance with MI-selected features:")\nprint(results_mi.to_string(index=False, float_format=\'%.4f\'))\n'




```python
# 4.2 Embedded Tree-based Feature Selection (ETB)
# Uses the feature importances from the best model to select significant features


def embedded_feature_selection(model, X, y, importance_threshold=0.001):
   """
   Select features based on importance scores from a tree-based model
   """
   importances = pd.Series(
       model.feature_importances_, 
       index=model.feature_names_in_
   ).sort_values(ascending=False)
   
   mask = model.feature_importances_ > importance_threshold
   selected_features = model.feature_names_in_[mask]
   X_selected = X[selected_features]
   
   print(f"Selected {len(selected_features)} features with importance > {importance_threshold}")
   return X_selected, importances

"""
# Perform ETB feature selection
X_train_etb, importance_scores = embedded_feature_selection(
   best_model_preprocessing, X_train_processed, Ytrain)

# Optimize hyperparameters for ETB-selected features
best_params_etb = optimize_brf(X_train_etb, Ytrain)
print("\nBest parameters for ETB-selected features:", best_params_etb)


# Evaluate performance
results_etb = get_model_performance(X_train_etb, Ytrain, best_params_etb)
print("\nModel Performance with ETB-selected features:")
print(results_etb.to_string(index=False, float_format='%.4f'))
"""
```




    '\n# Perform ETB feature selection\nX_train_etb, importance_scores = embedded_feature_selection(\n   best_model_preprocessing, X_train_processed, Ytrain)\n\n# Optimize hyperparameters for ETB-selected features\nbest_params_etb = optimize_brf(X_train_etb, Ytrain)\nprint("\nBest parameters for ETB-selected features:", best_params_etb)\n\n\n# Evaluate performance\nresults_etb = get_model_performance(X_train_etb, Ytrain, best_params_etb)\nprint("\nModel Performance with ETB-selected features:")\nprint(results_etb.to_string(index=False, float_format=\'%.4f\'))\n'




```python
# 4.3 RFECV
def rfecv_selection(X, y, cv=10):
   rfecv = RFECV(
       estimator=best_model_preprocessing,
       step=1,
       cv=cv,
       scoring="matthews_corrcoef"
   ).fit(X, y)
   
   selected_features = rfecv.get_feature_names_out()
   X_selected = X[selected_features]
   
   print(f"Optimal number of features: {rfecv.n_features_}")
   return X_selected, selected_features, rfecv

"""
# Run RFECV selection
X_train_rfecv, selected_features_rfecv, rfecv_selector = rfecv_selection(
   X_train_processed, Ytrain)

# Optimize hyperparameters
best_params_rfecv = optimize_brf(X_train_rfecv, Ytrain)
print("\nBest parameters for RFECV-selected features:", best_params_rfecv)


# Evaluate performance
results_rfecv = get_model_performance(X_train_rfecv, Ytrain, best_params_rfecv)
print("\nModel Performance with RFECV-selected features:")
print(results_rfecv.to_string(index=False, float_format='%.4f'))
"""
```




    '\n# Run RFECV selection\nX_train_rfecv, selected_features_rfecv, rfecv_selector = rfecv_selection(\n   X_train_processed, Ytrain)\n\n# Optimize hyperparameters\nbest_params_rfecv = optimize_brf(X_train_rfecv, Ytrain)\nprint("\nBest parameters for RFECV-selected features:", best_params_rfecv)\n\n\n# Evaluate performance\nresults_rfecv = get_model_performance(X_train_rfecv, Ytrain, best_params_rfecv)\nprint("\nModel Performance with RFECV-selected features:")\nprint(results_rfecv.to_string(index=False, float_format=\'%.4f\'))\n'




```python
#4.4 Genetic Algorithm Feature Selection (GA)

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import matthews_corrcoef, make_scorer
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Replace with your actual data
X_train_df = X_train_processed  # Your training feature DataFrame
y_train_df = Ytrain  # Your training label Series

# Save feature names
feature_names = X_train_df.columns

# Convert to numpy arrays
X_train = X_train_df.values
y_train = y_train_df.values

# Define evaluation function
def evaluate(individual):
   selected_features = [index for index, value in enumerate(individual) if value == 1]
   if not selected_features:
       return -1,  # Return -1 score if no features selected (worst case)
   X_train_selected = X_train[:, selected_features]
   clf = BalancedRandomForestClassifier(
       n_estimators=255,
       criterion="gini",
       max_depth=16,
       max_features=30,
       bootstrap=True,
       sampling_strategy='auto',  
       replacement=False,        
       random_state=1,
       verbose=False,
       n_jobs=-1
   )
   mcc_scorer = make_scorer(matthews_corrcoef)
   scores = cross_val_score(clf, X_train_selected, y_train, cv=5, scoring=mcc_scorer)
   return scores.mean(),

# Set GA parameters  
NUM_GENES = X_train.shape[1]  # Number of genes equals number of features
POP_SIZE = 50  # Population size
NGEN = 40     # Number of generations
CXPB = 0.5    # Crossover probability
MUTPB = 0.2   # Mutation probability

# Define genetic algorithm type
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_GENES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Elite preservation strategy
hof = tools.HallOfFame(1)

# Generate initial population
population = toolbox.population(n=POP_SIZE)

# Run genetic algorithm
result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                              stats=None, halloffame=hof, verbose=True)

# Get best individual
best_individual = hof[0]
selected_features = [index for index, value in enumerate(best_individual) if value == 1]

# Print selected feature names
selected_feature_names = feature_names[selected_features] 
print(f"Selected features: {selected_feature_names}")
print(f"Best individual MCC on training set: {evaluate(best_individual)[0]}")

# Train and evaluate model with selected features
X_train_selected = X_train[:, selected_features]
```

    gen	nevals
    0  	50    
    1  	26    
    2  	29    
    3  	28    
    4  	26    
    5  	32    
    6  	18    
    


```python
# """
# After getting selected features from GA, optimize model parameters and evaluate performance

# Optimize parameters using selected features
best_params_ga = optimize_brf(X_train_selected, Ytrain)
print("\nBest parameters with GA-selected features:")
print(best_params_ga)

# Evaluate model performance with selected features
results_ga = get_model_performance(X_train_selected, Ytrain, best_params_ga)
print("\nModel Performance with GA-selected features:")
print(results_ga.to_string(index=False, float_format='%.4f'))
# """
```




    '\nAfter getting selected features from GA, optimize model parameters and evaluate performance\n\n# Optimize parameters using selected features\nbest_params_ga = optimize_brf(X_train_selected, Ytrain)\nprint("\nBest parameters with GA-selected features:")\nprint(best_params_ga)\n\n# Evaluate model performance with selected features\nresults_ga = get_model_performance(X_train_selected, Ytrain, best_params_ga)\nprint("\nModel Performance with GA-selected features:")\nprint(results_ga.to_string(index=False, float_format=\'%.4f\'))\n'




```python
# Get selected features for testing set using feature names
X_test_selected = X_test_processed[selected_feature_names]
X_test_selected
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BalabanJ</th>
      <th>Chi0</th>
      <th>EState_VSA1</th>
      <th>EState_VSA10</th>
      <th>EState_VSA4</th>
      <th>EState_VSA6</th>
      <th>EState_VSA9</th>
      <th>HallKierAlpha</th>
      <th>Ipc</th>
      <th>Kappa3</th>
      <th>...</th>
      <th>fr_methoxy</th>
      <th>fr_morpholine</th>
      <th>fr_nitro_arom</th>
      <th>fr_para_hydroxylation</th>
      <th>fr_phos_ester</th>
      <th>fr_piperdine</th>
      <th>fr_pyridine</th>
      <th>fr_sulfide</th>
      <th>fr_term_acetylene</th>
      <th>fr_unbrch_alkane</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.927408</td>
      <td>0.460791</td>
      <td>-0.257990</td>
      <td>0.213746</td>
      <td>-0.101188</td>
      <td>-0.862098</td>
      <td>-0.990161</td>
      <td>0.723430</td>
      <td>-0.077297</td>
      <td>-0.064608</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.944306</td>
      <td>0.415761</td>
      <td>-0.278435</td>
      <td>0.004021</td>
      <td>1.979562</td>
      <td>-0.404055</td>
      <td>-0.511729</td>
      <td>0.184829</td>
      <td>-0.077298</td>
      <td>-0.060772</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>1.976799</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.838457</td>
      <td>2.908821</td>
      <td>3.180967</td>
      <td>2.176713</td>
      <td>1.388567</td>
      <td>0.642288</td>
      <td>2.116130</td>
      <td>0.621282</td>
      <td>0.264547</td>
      <td>-0.038419</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.370871</td>
      <td>-0.594226</td>
      <td>-0.533262</td>
      <td>-0.345554</td>
      <td>-1.087602</td>
      <td>2.344434</td>
      <td>0.084085</td>
      <td>-0.632359</td>
      <td>-0.077299</td>
      <td>-0.081032</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>1.976799</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.158339</td>
      <td>2.738372</td>
      <td>1.265508</td>
      <td>1.149556</td>
      <td>-0.150433</td>
      <td>0.054065</td>
      <td>1.672361</td>
      <td>-2.294595</td>
      <td>0.306693</td>
      <td>-0.061069</td>
      <td>...</td>
      <td>3.137389</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>3.862468</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>-0.893614</td>
      <td>1.136924</td>
      <td>1.999232</td>
      <td>2.024469</td>
      <td>-0.812459</td>
      <td>2.996690</td>
      <td>-0.990161</td>
      <td>-1.551696</td>
      <td>-0.077282</td>
      <td>-0.065891</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>3.862468</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>116</th>
      <td>0.753877</td>
      <td>-0.909708</td>
      <td>-0.791321</td>
      <td>-0.813410</td>
      <td>-0.143077</td>
      <td>1.054648</td>
      <td>-0.497209</td>
      <td>-0.437348</td>
      <td>-0.077299</td>
      <td>-0.092193</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>5.851491</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>117</th>
      <td>-0.665500</td>
      <td>-0.382062</td>
      <td>-0.791321</td>
      <td>-0.782961</td>
      <td>0.722049</td>
      <td>-0.862098</td>
      <td>-0.990161</td>
      <td>0.574851</td>
      <td>-0.077299</td>
      <td>-0.073523</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>-0.238492</td>
    </tr>
    <tr>
      <th>118</th>
      <td>0.648269</td>
      <td>0.592840</td>
      <td>0.203870</td>
      <td>0.641492</td>
      <td>-1.087602</td>
      <td>1.687195</td>
      <td>0.341147</td>
      <td>-1.170961</td>
      <td>-0.077298</td>
      <td>-0.056012</td>
      <td>...</td>
      <td>1.401270</td>
      <td>-0.102923</td>
      <td>7.656696</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>0.490743</td>
    </tr>
    <tr>
      <th>119</th>
      <td>0.945380</td>
      <td>-1.055985</td>
      <td>-0.791321</td>
      <td>-1.281364</td>
      <td>-0.352579</td>
      <td>-0.862098</td>
      <td>-0.990161</td>
      <td>1.522046</td>
      <td>-0.077299</td>
      <td>0.010308</td>
      <td>...</td>
      <td>-0.334849</td>
      <td>-0.102923</td>
      <td>-0.130605</td>
      <td>-0.345696</td>
      <td>-0.112867</td>
      <td>-0.307404</td>
      <td>-0.303219</td>
      <td>-0.222670</td>
      <td>-0.079556</td>
      <td>6.324624</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 65 columns</p>
</div>




```python
"""
5. Model Development and Evaluation

Note: RDKit_GA_65 represents the optimal molecular descriptor set. We constructed and optimized 
5 models using these selected features and validated them with an external test set. 
Due to the tedious nature of parameter optimization, we directly provide the optimal parameters 
for each machine learning algorithm based on our experimental results.

The following codes include:
1. Parameter optimization code templates for all 5 models:
  - Balanced Random Forest (BRF)
  - Easy Ensemble Classifier (EEC)
  - XGBoost with Balanced Bagging (BBC+XGBoost)
  - Gradient Boosting with Balanced Bagging (BBC+GBDT)
  - LightGBM with Balanced Bagging (BBC+LightGBM)

2. Model evaluation using best parameters:
  - 10-fold cross-validation performance
  - Out-of-fold performance
  - External test set performance
  - ROC curve comparison
"""
```




    '\n5. Model Development and Evaluation\n\nNote: RDKit_GA_65 represents the optimal molecular descriptor set. We constructed and optimized \n5 models using these selected features and validated them with an external test set. \nDue to the tedious nature of parameter optimization, we directly provide the optimal parameters \nfor each machine learning algorithm based on our experimental results.\n\nThe following codes include:\n1. Parameter optimization code templates for all 5 models:\n  - Balanced Random Forest (BRF)\n  - Easy Ensemble Classifier (EEC)\n  - XGBoost with Balanced Bagging (BBC+XGBoost)\n  - Gradient Boosting with Balanced Bagging (BBC+GBDT)\n  - LightGBM with Balanced Bagging (BBC+LightGBM)\n\n2. Model evaluation using best parameters:\n  - 10-fold cross-validation performance\n  - Out-of-fold performance\n  - External test set performance\n  - ROC curve comparison\n'




```python
def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
   """
   Evaluate model with CV, OOF and test set performance
   """
   def calculate_metrics(y_true, y_pred, y_pred_proba=None):
       tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
       metrics = {
           'ACC': (tp + tn) / (tp + tn + fp + fn),
           'SEN': tp / (tp + fn),
           'SPE': tn / (tn + fp),
           'MCC': matthews_corrcoef(y_true, y_pred)
       }
       if y_pred_proba is not None:
           metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
       return metrics

   # 10-fold CV evaluation
   cv = KFold(n_splits=10, shuffle=True, random_state=1)
   cv_metrics = []
   
   # For OOF predictions
   y_pred_all = np.zeros_like(y_train)
   y_pred_proba_all = np.zeros_like(y_train, dtype=float)
   
   # Get both CV and OOF predictions
   for train_idx, val_idx in cv.split(X_train):
       X_train_fold = X_train.iloc[train_idx]
       X_val_fold = X_train.iloc[val_idx]
       y_train_fold = y_train.iloc[train_idx]
       y_val_fold = y_train.iloc[val_idx]
       
       # Train model on fold
       model_fold = clone(model)
       model_fold.fit(X_train_fold, y_train_fold)
       
       # Get predictions for CV metrics
       y_pred = model_fold.predict(X_val_fold)
       y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
       
       # Store fold metrics
       fold_metrics = calculate_metrics(y_val_fold, y_pred, y_pred_proba)
       cv_metrics.append(fold_metrics)
       
       # Store OOF predictions
       y_pred_all[val_idx] = y_pred
       y_pred_proba_all[val_idx] = y_pred_proba

   # Calculate mean CV metrics
   cv_mean = {m: np.mean([fold[m] for fold in cv_metrics]) 
              for m in cv_metrics[0].keys()}
   
   # Calculate OOF metrics
   oof_metrics = calculate_metrics(y_train, y_pred_all, y_pred_proba_all)
   
   # Get test set predictions
   model.fit(X_train, y_train)
   y_pred_test = model.predict(X_test)
   y_pred_proba_test = model.predict_proba(X_test)[:, 1]
   test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
   
   # Create results DataFrame
   metrics_order = ['AUC', 'ACC', 'SEN', 'SPE', 'MCC']
   results = pd.DataFrame({
       'Metric': metrics_order,
       'CV Mean': [cv_mean[m] for m in metrics_order],
       'Out-of-fold': [oof_metrics[m] for m in metrics_order],
       'Test': [test_metrics[m] for m in metrics_order]
   })
   
   # Get confusion matrix for test set
   tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
   confusion_df = pd.DataFrame(
       {'Confusion_matrix': [tp, fn, fp, tn]},
       index=['TP', 'FN', 'FP', 'TN']
   )
   
   return results, confusion_df
```


```python
"""
Hyperparameter Optimization for Multiple Models

5.1. BalancedRandomForestClassifier Optimization
"""

import numpy as np
from sklearn.model_selection import KFold, cross_validate
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from imblearn.ensemble import BalancedRandomForestClassifier

def optimize_brf(X, y, random_state=1):
    """
    Optimize BalancedRandomForestClassifier hyperparameters
    """
    def objective(params):
        model = BalancedRandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            criterion=params["criterion"],
            max_depth=int(params["max_depth"]),
            max_features=params["max_features"],
            sampling_strategy='auto',
            replacement=False,
            random_state=random_state,
            n_jobs=-1
        )
        cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring="matthews_corrcoef",
            n_jobs=-1,
            error_score="raise"
        )
        return -np.mean(scores["test_score"])

    # Define parameter space
    max_features_choices = ["sqrt", "log2"] + list(range(10, 70))
    param_space = {
        'n_estimators': hp.quniform("n_estimators", 40, 300, 1),
        'criterion': hp.choice("criterion", ["gini", "entropy"]),
        'max_depth': hp.quniform('max_depth', 1, 25, 1),
        'max_features': hp.choice("max_features", max_features_choices)
    }
    
    # Run optimization
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
        early_stop_fn=no_progress_loss(400),
        verbose=True
    )
    
    # Convert parameters to correct types
    best_params = {
        "n_estimators": int(best_params["n_estimators"]),
        "criterion": ["gini", "entropy"][best_params["criterion"]],
        "max_depth": int(best_params["max_depth"]),
        "max_features": max_features_choices[best_params["max_features"]],
        "bootstrap": True,
        "sampling_strategy": 'auto',
        "replacement": False,
        "random_state": random_state,
        "verbose": False,
        "n_jobs": -1
    }
    
    return best_params

# Example usage:
"""
# Optimize BRF parameters
best_params_brf = optimize_brf(X_train_selected_df, Ytrain)
print("\nBest parameters for BRF:", best_params_brf)

# Initialize and train model with best parameters
model_brf = BalancedRandomForestClassifier(**best_params_brf)
model_brf.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_brf, confusion_matrix_brf = evaluate_model_performance(
   model_brf, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nBalanced Random Forest Performance:")
print(results_brf.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_brf)

"""
```




    '\n# Optimize BRF parameters\nbest_params_brf = optimize_brf(X_train_selected_df, Ytrain)\nprint("\nBest parameters for BRF:", best_params_brf)\n\n# Initialize and train model with best parameters\nmodel_brf = BalancedRandomForestClassifier(**best_params_brf)\nmodel_brf.fit(X_train_selected_df, Ytrain)\n\n# Evaluate performance\nresults_brf, confusion_matrix_brf = evaluate_model_performance(\n   model_brf, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)\n\nprint("\nBalanced Random Forest Performance:")\nprint(results_brf.to_string(index=False, float_format=\'%.4f\'))\nprint("\nConfusion Matrix:")\nprint(confusion_matrix_brf)\n\n'




```python
"""
5.1 BalancedRandomForestClassifier

Using optimized parameters to evaluate model performance
"""

from sklearn.base import clone
from sklearn.model_selection import KFold

# Best model parameters from GA optimization
best_params_ga = {
   'n_estimators': 154,
   'criterion': "gini",  
   'max_depth': 15,
   'max_features': 48,
   'sampling_strategy': 'auto',
   'replacement': False, 
   'random_state': 1,
   'bootstrap': True,
   'verbose': False,
   'n_jobs': -1
}

# Convert numpy arrays back to DataFrames for selected features
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names) 

# Initialize and train model
model_brf = BalancedRandomForestClassifier(**best_params_ga)
model_brf.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_brf, confusion_matrix_brf = evaluate_model_performance(
   model_brf, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nBalanced Random Forest Performance:")
print(results_brf.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:") 
print(confusion_matrix_brf)
```

    
    Balanced Random Forest Performance:
    Metric  CV Mean  Out-of-fold   Test
       AUC   0.8749       0.8764 0.8887
       ACC   0.8177       0.8176 0.8083
       SEN   0.8194       0.8305 0.7667
       SPE   0.8153       0.8134 0.8222
       MCC   0.5863       0.5841 0.5444
    
    Confusion Matrix:
        Confusion_matrix
    TP                23
    FN                 7
    FP                16
    TN                74
    


```python
"""
5.2. EasyEnsembleClassifier Optimization and Evaluation
"""

from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import KFold, cross_validate
import numpy as np

def optimize_eec(X, y, random_state=1):
   """
   Optimize EasyEnsembleClassifier hyperparameters using hyperopt
   """
   # Define parameter space
   param_space = {
       'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
       'learning_rate': hp.quniform('learning_rate', 0.2, 1.0, 0.1), 
       'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
       'base_estimator_max_depth': hp.quniform('base_estimator_max_depth', 1, 10, 1)
   }

   def objective(params):
       base_estimator = DecisionTreeClassifier(
           max_depth=int(params['base_estimator_max_depth']), 
           random_state=random_state
       )
       
       model = EasyEnsembleClassifier(
           n_estimators=10,
           estimator=AdaBoostClassifier(
               estimator=base_estimator,
               n_estimators=int(params['n_estimators']),
               learning_rate=params['learning_rate'],
               algorithm=params['algorithm'],
               random_state=random_state
           ),
           random_state=random_state,
           n_jobs=-1
       )
       
       cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
       scores = cross_validate(
           model, X, y,
           cv=cv,
           scoring="matthews_corrcoef",
           n_jobs=-1,
           error_score="raise"
       )
       return -np.mean(scores["test_score"])

   # Run optimization
   trials = Trials()
   best_params = fmin(
       fn=objective,
       space=param_space,
       algo=tpe.suggest,
       max_evals=200,
       trials=trials,
       early_stop_fn=no_progress_loss(100),
       verbose=True
   )

   # Convert parameters
   algorithms = ['SAMME', 'SAMME.R']
   best_params = {
       'n_estimators':10,
       'estimator': AdaBoostClassifier(
           estimator=DecisionTreeClassifier(
               max_depth=int(best_params['base_estimator_max_depth']),
               random_state=random_state
           ),
           n_estimators=int(best_params['n_estimators']),
           learning_rate=best_params['learning_rate'],
           algorithm=algorithms[best_params['algorithm']],
           random_state=random_state
       ),
       'random_state': random_state,
       'n_jobs': -1
   }
   
   return best_params

# Example usage:
"""
# Optimize parameters
print("Optimizing EasyEnsemble parameters...")
best_params_eec = optimize_eec(X_train_selected_df, Ytrain)
print("\nBest parameters:", best_params_eec)

# Initialize and train model with best parameters
model_eec = EasyEnsembleClassifier(**best_params_eec)
model_eec.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_eec, confusion_matrix_eec = evaluate_model_performance(
   model_eec, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nEasyEnsemble Classifier Performance:")
print(results_eec.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_eec)

"""
```




    '\n# Optimize parameters\nprint("Optimizing EasyEnsemble parameters...")\nbest_params_eec = optimize_eec(X_train_selected_df, Ytrain)\nprint("\nBest parameters:", best_params_eec)\n\n# Initialize and train model with best parameters\nmodel_eec = EasyEnsembleClassifier(**best_params_eec)\nmodel_eec.fit(X_train_selected_df, Ytrain)\n\n# Evaluate performance\nresults_eec, confusion_matrix_eec = evaluate_model_performance(\n   model_eec, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)\n\nprint("\nEasyEnsemble Classifier Performance:")\nprint(results_eec.to_string(index=False, float_format=\'%.4f\'))\nprint("\nConfusion Matrix:")\nprint(confusion_matrix_eec)\n\n'




```python
"""
5.2. EasyEnsembleClassifier

Using optimized parameters to evaluate model performance
"""

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier

# Initialize EasyEnsemble with best parameters
model_eec = EasyEnsembleClassifier(
   n_estimators=10,
   estimator=AdaBoostClassifier(
       estimator=DecisionTreeClassifier(max_depth=7, random_state=1),
       n_estimators=178,
       learning_rate=0.92,
       algorithm='SAMME.R',
       random_state=1
   ),
   random_state=1,
   n_jobs=-1
)

# Convert numpy arrays back to DataFrames for selected features
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)

# Train model
model_eec.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_eec, confusion_matrix_eec = evaluate_model_performance(
   model_eec, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nEasyEnsemble Classifier Performance:")
print(results_eec.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_eec)
```

    
    EasyEnsemble Classifier Performance:
    Metric  CV Mean  Out-of-fold   Test
       AUC   0.8805       0.8836 0.8930
       ACC   0.8283       0.8281 0.8500
       SEN   0.8146       0.8220 0.8333
       SPE   0.8302       0.8301 0.8556
       MCC   0.5971       0.5978 0.6413
    
    Confusion Matrix:
        Confusion_matrix
    TP                25
    FN                 5
    FP                13
    TN                77
    


```python
"""
5.3. XGBoost with BalancedBagging Optimization and Evaluation
"""

def optimize_xgb(X, y, random_state=1):
   """
   Optimize XGBoost with BalancedBagging hyperparameters
   """
   # Calculate scale_pos_weight
   n_positive = np.sum(y == 1)
   n_negative = np.sum(y == 0) 
   scale_pos_weight = n_negative / n_positive
   
   def objective(params):
       booster_options = ["gbtree", "gblinear", "dart"]
       booster = booster_options[params["booster"]]
       
       base_model = XGBC(
           n_estimators=int(params["n_estimators"]),
           learning_rate=params["learning_rate"],
           booster=booster,
           colsample_bytree=params["colsample_bytree"],
           colsample_bynode=params["colsample_bynode"],
           gamma=params["gamma"],
           reg_lambda=params["reg_lambda"],
           min_child_weight=int(params["min_child_weight"]),
           max_depth=int(params["max_depth"]),
           subsample=params["subsample"],
           scale_pos_weight=scale_pos_weight,
           random_state=random_state,
           verbosity=0,
           n_jobs=-1
       )
       
       model = BalancedBaggingClassifier(
           n_estimators=10,
           estimator=base_model,
           random_state=random_state
       )
       
       cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
       scores = cross_validate(
           model, X, y,
           cv=cv,
           scoring="matthews_corrcoef",
           n_jobs=-1,
           error_score="raise"
       )
       return -np.mean(scores["test_score"])

   # Define parameter space
   param_space = {
       'n_estimators': hp.quniform("n_estimators", 20, 300, 1),
       'learning_rate': hp.quniform("learning_rate", 0.1, 1.0, 0.1),
       'booster': hp.choice("booster", [0, 1, 2]),
       'colsample_bytree': hp.quniform("colsample_bytree", 0.3, 1, 0.1),
       'colsample_bynode': hp.quniform("colsample_bynode", 0.3, 1, 0.1),
       'gamma': hp.loguniform('gamma', -5, 0),
       'reg_lambda': hp.loguniform("reg_lambda", -5, 0),
       'min_child_weight': hp.quniform("min_child_weight", 1, 10, 1),
       'max_depth': hp.quniform("max_depth", 2, 20, 1),
       'subsample': hp.quniform("subsample", 0.3, 1, 0.1)
   }
   
   # Run optimization
   trials = Trials()
   best_params = fmin(
       fn=objective,
       space=param_space,
       algo=tpe.suggest,
       max_evals=200,
       trials=trials,
       early_stop_fn=no_progress_loss(400),
       verbose=True
   )
   
   # Convert parameters to final model form
   booster_options = ["gbtree", "gblinear", "dart"]
   best_params = {
       'estimator': XGBC(
           n_estimators=int(best_params["n_estimators"]),
           learning_rate=best_params["learning_rate"],
           booster=booster_options[best_params["booster"]],
           colsample_bytree=best_params["colsample_bytree"],
           colsample_bynode=best_params["colsample_bynode"],
           gamma=best_params["gamma"],
           reg_lambda=best_params["reg_lambda"],
           min_child_weight=int(best_params["min_child_weight"]),
           max_depth=int(best_params["max_depth"]),
           subsample=best_params["subsample"],
           scale_pos_weight=scale_pos_weight,
           random_state=random_state,
           verbosity=0,
           n_jobs=-1
       ),
       'n_estimators': 10,
       'random_state': random_state
   }
   
   return best_params

# Example usage:
"""

# Optimize parameters
print("Optimizing XGBoost parameters...")
best_params_xgb = optimize_xgb(X_train_selected_df, Ytrain)
print("\nBest parameters:", best_params_xgb)

# Initialize and train model with best parameters
model_xgb = BalancedBaggingClassifier(**best_params_xgb)
model_xgb.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_xgb, confusion_matrix_xgb = evaluate_model_performance(
   model_xgb, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nXGBoost with BalancedBagging Performance:")
print(results_xgb.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_xgb)

"""
```




    '\n\n# Optimize parameters\nprint("Optimizing XGBoost parameters...")\nbest_params_xgb = optimize_xgb(X_train_selected_df, Ytrain)\nprint("\nBest parameters:", best_params_xgb)\n\n# Initialize and train model with best parameters\nmodel_xgb = BalancedBaggingClassifier(**best_params_xgb)\nmodel_xgb.fit(X_train_selected_df, Ytrain)\n\n# Evaluate performance\nresults_xgb, confusion_matrix_xgb = evaluate_model_performance(\n   model_xgb, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)\n\nprint("\nXGBoost with BalancedBagging Performance:")\nprint(results_xgb.to_string(index=False, float_format=\'%.4f\'))\nprint("\nConfusion Matrix:")\nprint(confusion_matrix_xgb)\n\n'




```python
"""
5.3. XGBoost with BalancedBagging

Using optimized parameters to evaluate model performance
"""

from xgboost import XGBClassifier as XGBC
from imblearn.ensemble import BalancedBaggingClassifier

# Calculate scale_pos_weight
n_positive = np.sum(Ytrain == 1)
n_negative = np.sum(Ytrain == 0)
scale_pos_weight = n_negative / n_positive

# Define booster options
booster_options = {0: 'gbtree', 1: 'gblinear', 2: 'dart'}

# Initialize base XGBoost model
xgb = XGBC(
   n_estimators=172,
   learning_rate=0.73,
   booster=booster_options[2],  # 'dart'
   colsample_bytree=0.3,
   colsample_bynode=1.0,
   gamma=0.036296772856035525,
   reg_lambda=0.06781903189364931,
   min_child_weight=1.0,
   max_depth=18,
   subsample=0.9,
   scale_pos_weight=scale_pos_weight,
   random_state=1,
   verbosity=0,
   n_jobs=-1
)

# Initialize BalancedBagging with XGBoost
model_xgb = BalancedBaggingClassifier(
   n_estimators=10,
   random_state=1,
   estimator=xgb
)

# Convert numpy arrays back to DataFrames for selected features
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)

# Train model
model_xgb.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_xgb, confusion_matrix_xgb = evaluate_model_performance(
   model_xgb, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nXGBoost with BalancedBagging Performance:")
print(results_xgb.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_xgb)
```

    
    XGBoost with BalancedBagging Performance:
    Metric  CV Mean  Out-of-fold   Test
       AUC   0.8442       0.8499 0.8763
       ACC   0.7985       0.7987 0.8083
       SEN   0.7725       0.7797 0.7333
       SPE   0.8033       0.8050 0.8333
       MCC   0.5275       0.5327 0.5313
    
    Confusion Matrix:
        Confusion_matrix
    TP                22
    FN                 8
    FP                15
    TN                75
    


```python
"""
5.4. Gradient Boosting with BalancedBagging Optimization and Evaluation
"""

def optimize_gbc(X, y, random_state=1):
   """
   Optimize Gradient Boosting with BalancedBagging hyperparameters
   """
   def objective(params):
       base_model = GBC(
           n_estimators=int(params["n_estimators"]),
           learning_rate=params["learning_rate"],
           criterion=params["criterion"],
           max_depth=int(params["max_depth"]),
           max_features=params["max_features"],
           subsample=params["subsample"],
           random_state=random_state,
           verbose=False
       )
       
       model = BalancedBaggingClassifier(
           n_estimators=10,
           estimator=base_model,
           random_state=random_state
       )
       
       cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
       scores = cross_validate(
           model, X, y,
           cv=cv,
           scoring="matthews_corrcoef",
           n_jobs=-1,
           error_score="raise"
       )
       return -np.mean(scores["test_score"])

   # Define parameter space
   max_features_choices = ["sqrt", "log2"] + list(range(2, 60, 1))
   param_space = {
       'n_estimators': hp.quniform("n_estimators", 25, 250, 1),
       'learning_rate': hp.quniform("learning_rate", 0.1, 1.0, 0.1),
       'criterion': hp.choice("criterion", ["friedman_mse", "squared_error"]),
       'max_depth': hp.quniform("max_depth", 2, 10, 1),
       'subsample': hp.quniform("subsample", 0.5, 1.0, 0.1),
       'max_features': hp.choice("max_features", max_features_choices)
   }
   
   # Run optimization
   trials = Trials()
   best_params = fmin(
       fn=objective,
       space=param_space,
       algo=tpe.suggest,
       max_evals=300,
       trials=trials,
       early_stop_fn=no_progress_loss(400),
       verbose=True
   )
   
   # Convert parameters to final model form
   criterion_choices = ["friedman_mse", "squared_error"]
   best_params = {
       'estimator': GBC(
           n_estimators=int(best_params["n_estimators"]),
           learning_rate=best_params["learning_rate"],
           criterion=criterion_choices[best_params["criterion"]],
           max_depth=int(best_params["max_depth"]),
           max_features=max_features_choices[best_params["max_features"]],
           subsample=best_params["subsample"],
           random_state=random_state,
           verbose=False
       ),
       'n_estimators': 10,
       'random_state': random_state
   }
   
   return best_params

# Example usage:
"""

# Optimize parameters
print("Optimizing Gradient Boosting parameters...")
best_params_gbc = optimize_gbc(X_train_selected_df, Ytrain)
print("\nBest parameters:", best_params_gbc)

# Initialize and train model with best parameters
model_gbc = BalancedBaggingClassifier(**best_params_gbc)
model_gbc.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_gbc, confusion_matrix_gbc = evaluate_model_performance(
   model_gbc, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nGradient Boosting with BalancedBagging Performance:")
print(results_gbc.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_gbc)

"""
```




    '\n\n# Optimize parameters\nprint("Optimizing Gradient Boosting parameters...")\nbest_params_gbc = optimize_gbc(X_train_selected_df, Ytrain)\nprint("\nBest parameters:", best_params_gbc)\n\n# Initialize and train model with best parameters\nmodel_gbc = BalancedBaggingClassifier(**best_params_gbc)\nmodel_gbc.fit(X_train_selected_df, Ytrain)\n\n# Evaluate performance\nresults_gbc, confusion_matrix_gbc = evaluate_model_performance(\n   model_gbc, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)\n\nprint("\nGradient Boosting with BalancedBagging Performance:")\nprint(results_gbc.to_string(index=False, float_format=\'%.4f\'))\nprint("\nConfusion Matrix:")\nprint(confusion_matrix_gbc)\n\n'




```python
"""
5.4. Gradient Boosting with BalancedBagging

Using optimized parameters to evaluate model performance
"""

from sklearn.ensemble import GradientBoostingClassifier as GBC
from imblearn.ensemble import BalancedBaggingClassifier

# Initialize GBC model with best parameters
gbc = GBC(
   n_estimators=107,
   learning_rate=0.24,
   criterion='friedman_mse',
   max_depth=5,
   max_features=4,
   subsample=0.99,
   random_state=1,
   verbose=False
)

# Wrap with BalancedBaggingClassifier
model_gbc = BalancedBaggingClassifier(
   n_estimators=10,
   random_state=1,
   estimator=gbc
)

# Convert numpy arrays back to DataFrames for selected features
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)

# Train model
model_gbc.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_gbc, confusion_matrix_gbc = evaluate_model_performance(
   model_gbc, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nGradient Boosting with BalancedBagging Performance:")
print(results_gbc.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_gbc)
```

    
    Gradient Boosting with BalancedBagging Performance:
    Metric  CV Mean  Out-of-fold   Test
       AUC   0.8796       0.8773 0.8974
       ACC   0.8363       0.8365 0.8500
       SEN   0.7416       0.7458 0.7333
       SPE   0.8667       0.8663 0.8889
       MCC   0.5863       0.5850 0.6093
    
    Confusion Matrix:
        Confusion_matrix
    TP                22
    FN                 8
    FP                10
    TN                80
    


```python
"""
5.5. LightGBM with BalancedBagging Optimization and Evaluation
"""

def optimize_lgbm(X, y, random_state=1):
   """
   Optimize LightGBM with BalancedBagging hyperparameters
   """
   def objective(params):
       base_model = LGBMClassifier(
           boosting_type=params['boosting_type'],
           n_estimators=int(params['n_estimators']),
           learning_rate=params['learning_rate'], 
           max_depth=int(params['max_depth']),
           num_leaves=int(params['num_leaves']),
           colsample_bytree=params['colsample_bytree'],
           subsample=params['subsample'],
           reg_alpha=params['reg_alpha'],
           reg_lambda=params['reg_lambda'],
           min_child_samples=int(params["min_child_samples"]),
           random_state=random_state,
           class_weight="balanced",
           n_jobs=-1
       )
       
       model = BalancedBaggingClassifier(
           n_estimators=10,
           estimator=base_model,
           random_state=random_state
       )
       
       cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
       scores = cross_validate(
           model, X, y,
           cv=cv,
           scoring="matthews_corrcoef",
           n_jobs=-1,
           error_score="raise"
       )
       return -np.mean(scores["test_score"])

   # Define parameter space
   param_space = {
       'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
       'n_estimators': hp.quniform('n_estimators', 40, 300, 1),
       'learning_rate': hp.quniform("learning_rate", 0.01, 1.0, 0.01),
       'max_depth': hp.quniform('max_depth', 2, 30, 1),
       'num_leaves': hp.quniform('num_leaves', 10, 100, 1),
       'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1.0, 0.01),
       'subsample': hp.quniform('subsample', 0.5, 1.0, 0.01),
       'reg_alpha': hp.loguniform('reg_alpha', -5, 0),
       'reg_lambda': hp.loguniform('reg_lambda', -5, 0),
       'min_child_samples': hp.quniform('min_child_samples', 1, 20, 1)
   }
   
   # Run optimization
   trials = Trials()
   best_params = fmin(
       fn=objective,
       space=param_space,
       algo=tpe.suggest,
       max_evals=300,
       trials=trials,
       early_stop_fn=no_progress_loss(400),
       verbose=True
   )
   
   # Convert parameters to final model form
   boosting_types = ['gbdt', 'dart']
   best_params = {
       'estimator': LGBMClassifier(
           boosting_type=boosting_types[best_params['boosting_type']],
           n_estimators=int(best_params['n_estimators']),
           learning_rate=best_params['learning_rate'],
           max_depth=int(best_params['max_depth']),
           num_leaves=int(best_params['num_leaves']),
           colsample_bytree=best_params['colsample_bytree'],
           subsample=best_params['subsample'],
           reg_alpha=best_params['reg_alpha'],
           reg_lambda=best_params['reg_lambda'],
           min_child_samples=int(best_params['min_child_samples']),
           random_state=random_state,
           class_weight="balanced",
           n_jobs=-1
       ),
       'n_estimators': 10,
       'random_state': random_state
   }
   
   return best_params

# Example usage:
"""

# Optimize parameters
print("Optimizing LightGBM parameters...")
best_params_lgbm = optimize_lgbm(X_train_selected_df, Ytrain)
print("\nBest parameters:", best_params_lgbm)

# Initialize and train model with best parameters
model_lgbm = BalancedBaggingClassifier(**best_params_lgbm)
model_lgbm.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_lgbm, confusion_matrix_lgbm = evaluate_model_performance(
   model_lgbm, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nLightGBM with BalancedBagging Performance:")
print(results_lgbm.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_lgbm)

"""
```




    '\n\n# Optimize parameters\nprint("Optimizing LightGBM parameters...")\nbest_params_lgbm = optimize_lgbm(X_train_selected_df, Ytrain)\nprint("\nBest parameters:", best_params_lgbm)\n\n# Initialize and train model with best parameters\nmodel_lgbm = BalancedBaggingClassifier(**best_params_lgbm)\nmodel_lgbm.fit(X_train_selected_df, Ytrain)\n\n# Evaluate performance\nresults_lgbm, confusion_matrix_lgbm = evaluate_model_performance(\n   model_lgbm, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)\n\nprint("\nLightGBM with BalancedBagging Performance:")\nprint(results_lgbm.to_string(index=False, float_format=\'%.4f\'))\nprint("\nConfusion Matrix:")\nprint(confusion_matrix_lgbm)\n\n'




```python
"""
5.5. LightGBM with BalancedBagging using GA-selected features
"""

from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedBaggingClassifier

# Initialize LightGBM model with best parameters
lgbm = LGBMClassifier(
   boosting_type='gbdt',
   n_estimators=112,
   learning_rate=0.83,
   max_depth=14,
   num_leaves=85,
   colsample_bytree=0.55,
   subsample=0.83,
   min_child_samples=2,
   reg_alpha=0.011600450241817575,
   reg_lambda=0.12670847895140583,
   random_state=1,
   class_weight="balanced",
   n_jobs=-1
)

# Wrap with BalancedBaggingClassifier
model_lgbm = BalancedBaggingClassifier(
   n_estimators=10,
   random_state=1,
   estimator=lgbm
)

# Convert numpy arrays back to DataFrames for selected features
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)

# Train model
model_lgbm.fit(X_train_selected_df, Ytrain)

# Evaluate performance
results_lgbm, confusion_matrix_lgbm = evaluate_model_performance(
   model_lgbm, X_train_selected_df, Ytrain, X_test_selected_df, Ytest)

print("\nLightGBM with BalancedBagging Performance:")
print(results_lgbm.to_string(index=False, float_format='%.4f'))
print("\nConfusion Matrix:")
print(confusion_matrix_lgbm)
```

    
    LightGBM with BalancedBagging Performance:
    Metric  CV Mean  Out-of-fold   Test
       AUC   0.8654       0.8653 0.8915
       ACC   0.8279       0.8281 0.8417
       SEN   0.7416       0.7458 0.7333
       SPE   0.8546       0.8552 0.8778
       MCC   0.5697       0.5694 0.5926
    
    Confusion Matrix:
        Confusion_matrix
    TP                22
    FN                 8
    FP                11
    TN                79
    


```python
"""
Model Comparison: ROC Curves for All Models
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib import rcParams

def plot_roc_curves(models, X_test, y_test, title="ROC Curves Comparison", save_path=None):
   """
   Plot ROC curves for multiple models
   
   Args:
       models: Dictionary of model name and fitted model object
       X_test: Test features
       y_test: Test labels
       title: Plot title
       save_path: Path to save the plot
   """
   # Set global font
   rcParams['font.family'] = 'Times New Roman'
   
   # Calculate ROC curves for each model
   roc_data = {}
   for name, model in models.items():
       y_pred_proba = model.predict_proba(X_test)[:, 1]
       fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
       roc_auc = auc(fpr, tpr)
       roc_data[name] = {
           'fpr': fpr,
           'tpr': tpr,
           'auc': roc_auc
       }
   
   # Sort models by AUC score
   sorted_models = sorted(roc_data.items(), key=lambda x: x[1]['auc'], reverse=True)
   
   # Define colors
   colors = ['darkorange', 'purple', 'red', 'green', 'blue']
   
   # Create plot
   plt.figure(figsize=(10, 8))
   
   # Plot ROC curve for each model
   for i, (name, data) in enumerate(sorted_models):
       plt.plot(
           data['fpr'],
           data['tpr'],
           color=colors[i],
           lw=2,
           label=f'{name} (AUC = {data["auc"]:.4f})'
       )
   
   # Plot random baseline
   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
   
   # Set plot parameters
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.0])
   plt.xlabel('False Positive Rate', fontsize=22, fontweight='bold')
   plt.ylabel('True Positive Rate', fontsize=22, fontweight='bold')
   plt.xticks(fontsize=22, fontweight='bold')
   plt.yticks(fontsize=22, fontweight='bold')
   plt.legend(loc="lower right", prop={'size': 18, 'weight': 'bold'})
   plt.title(title, fontsize=22, fontweight='bold')
   
   # Save plot
   if save_path:
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
   plt.show()

# Create dictionary of models
models = {
   'BRF': model_brf,
   'EEC': model_eec,
   'BBC+XGBoost': model_xgb,
   'BBC+GBDT': model_gbc,
   'BBC+LightGBM': model_lgbm
}

# Plot ROC curves
plot_roc_curves(
   models, 
   X_test_selected_df, 
   Ytest,
   title='RDKit_GA_65',
   save_path='roc_curve_RDKit_GA_65.png'
)
```


    
![png](DIA_RDKit_Prediction_files/DIA_RDKit_Prediction_33_0.png)
    



```python
"""
6.Model Interpretability Analysis Based on Best Model (EasyEnsemble Classifier)
Using SHAP (SHapley Additive exPlanations) to analyze feature importance

Note: Due to the long computation time of SHAP values, we will load pre-computed 
SHAP values to demonstrate the visualization of interpretability analysis.
"""

import shap

# Create SHAP explainer
explainer = shap.KernelExplainer(
   best_model_eec.predict_proba, 
   X_train_selected_df
)

# Calculate SHAP values for training set
shap_values = explainer.shap_values(
   X_train_selected_df,  
   nsamples=477
)
```


```python
"""
6.1 Load pre-computed SHAP values and create visualization of top 10 features
Combining bee swarm plot and feature importance bar plot with customized styling
"""

import matplotlib.pyplot as plt
import shap
import numpy as np
import pickle


# Load pre-computed SHAP values
with open('shap_values_train.pkl', 'rb') as f:
   shap_values = pickle.load(f)

# Get feature names from selected features
feature_names = selected_feature_names  # Using previously stored feature names

# Select SHAP values for positive class (class 1)
shap_values_class_1 = shap_values[:, :, 1]


# Calculate feature importance and get top 10 features
feature_importance = np.mean(np.abs(shap_values_class_1), axis=0)
top10_indices = np.argsort(feature_importance)[-10:][::-1]
top10_features = selected_feature_names[top10_indices]
shap_values_class_1_top10 = shap_values_class_1[:, top10_indices]
X_train_selected_top10 = X_train_selected_df.iloc[:, top10_indices]

# Create main figure
fig, ax1 = plt.subplots(figsize=(10, 4), dpi=300)

# Draw bee swarm plot with color bar
shap.summary_plot(
   shap_values_class_1_top10,
   X_train_selected_top10,
   feature_names=top10_features,
   plot_type="dot",
   cmap='coolwarm',
   show=False,
   color_bar=True
)

# Adjust plot position
plt.gca().set_position([0.2, 0.4, 0.5, 0.65])

# Get shared y-axis and create second axis
ax1 = plt.gca()
ax2 = ax1.twiny()

# Draw feature importance bar plot
shap.summary_plot(
   shap_values_class_1_top10,
   X_train_selected_top10,
   feature_names=top10_features,
   plot_type="bar",
   show=False
)
plt.gca().set_position([0.2, 0.3, 0.5, 0.65])

# Add horizontal line and adjust bar transparency
ax2.axhline(y=10, color='gray', linestyle='-', linewidth=1)
for bar in ax2.patches:
   bar.set_alpha(0.2)

# Set axis labels and properties
ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
ax1.set_ylabel('Features', fontsize=12)

# Adjust tick parameters
ax1.tick_params(axis='both', labelsize=16)
ax2.tick_params(axis='both', labelsize=16)

# Position top x-axis
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()

# Adjust legends if present
if ax1.get_legend():
   ax1.legend(prop={'size': 12})
if ax2.get_legend():
   ax2.legend(prop={'size': 12})

plt.tight_layout()

# Save high-resolution plot
plt.savefig("SHAP_combined_with_top_line_corrected_train_top10.png",
           format='png',
           bbox_inches='tight')
plt.show()
```


    
![png](DIA_RDKit_Prediction_files/DIA_RDKit_Prediction_35_0.png)
    



```python
"""
6.2 SHAP Dependence Plot Analysis
Creating dependence plots for top 10 features with original scale values
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import fsolve
import joblib
from sklearn.preprocessing import StandardScaler
from typing import List, Union
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')

# 1. First standardize the original features
scaler = StandardScaler()
X_train_std = pd.DataFrame(
    scaler.fit_transform(Xtrain),
    columns=Xtrain.columns,
    index=Xtrain.index
)

# 2. Get original feature names
original_feature_names = list(Xtrain.columns)

def plot_shap_dependence(
    feature_list: List[str],
    X_train_selected: np.ndarray,
    shap_values_array: np.ndarray,
    scaler: StandardScaler,
    feature_names: Union[List[str], pd.Index],
    original_feature_names: List[str],
    file_name: str = "SHAP_Dependence_Plots.png",
    fig_size: tuple = (25, 10),
    font_size: int = 22,
    dpi: int = 600
) -> None:
    """
    Create SHAP dependence plots for specified features, showing original (unscaled) feature values.
    """
    # Convert feature_names to list if it's an Index
    feature_names = list(feature_names)
    
    # Create subplot grid
    fig, axs = plt.subplots(2, 5, figsize=fig_size, dpi=dpi)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Process each feature
    for i, feature in enumerate(feature_list):
        # Get current subplot
        ax = axs[i // 5, i % 5]
        
        # Get the index of the feature in both selected and original features
        selected_idx = feature_names.index(feature)
        original_idx = original_feature_names.index(feature)
        
        # Get scaled feature values
        scaled_values = X_train_selected[:, selected_idx]
        
        # Create dummy array with size of original features
        dummy_X = np.zeros((len(scaled_values), len(original_feature_names)))
        dummy_X[:, original_idx] = scaled_values
        
        # Convert to original scale using the scaler
        original_feature_values = scaler.inverse_transform(dummy_X)[:, original_idx]
        feature_shap_values = shap_values_array[:, selected_idx]
        
        # Create scatter plot
        ax.scatter(
            original_feature_values,
            feature_shap_values,
            s=10,
            color='deepskyblue',
            alpha=0.6
        )
        
        # Add LOWESS regression line
        lowess_plot = sns.regplot(
            x=original_feature_values,
            y=feature_shap_values,
            scatter=False,
            lowess=True,
            color='lightcoral',
            ax=ax
        )
        
        # Get LOWESS line data
        line = lowess_plot.get_lines()[0]
        x_fit = line.get_xdata()
        y_fit = line.get_ydata()
        
        # Find zero crossings
        def find_zero_crossings(x_fit, y_fit):
            crossings = []
            for j in range(1, len(y_fit)):
                if (y_fit[j-1] < 0 and y_fit[j] > 0) or (y_fit[j-1] > 0 and y_fit[j] < 0):
                    if feature == 'NumAliphaticRings':
                        crossings.append(3)
                    else:
                        crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x_fit[j])[0]
                        crossings.append(crossing)
            return crossings
        
        # Add zero crossing lines and annotations
        x_intercepts = find_zero_crossings(x_fit, y_fit)
        for x_intercept in x_intercepts:
            ax.axvline(x=x_intercept, color='blue', linestyle='--', alpha=0.5)
            ax.text(
                x_intercept,
                0.01,
                f'{x_intercept:.2f}',
                color='black',
                fontsize=font_size,
                verticalalignment='bottom',
                fontweight='bold',
                fontname='Times New Roman'
            )
        
        # Add horizontal zero line
        ax.axhline(y=0, color='black', linestyle='-.', linewidth=1, alpha=0.5)
        
        # Set labels
        ax.set_xlabel(
            feature,
            fontsize=font_size,
            fontweight='bold',
            fontname='Times New Roman'
        )
        ax.set_ylabel(
            'SHAP value',
            fontsize=font_size,
            fontweight='bold',
            fontname='Times New Roman'
        )
        
        # Style tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
            label.set_fontsize(font_size)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Save and display plot
    plt.savefig(file_name, format='png', bbox_inches='tight')
    plt.show()

# Use the function
# Your top 10 features for visualization
feature_list = [
    'SlogP_VSA10', 'PEOE_VSA9', 'SMR_VSA10', 'EState_VSA1', 'NumAliphaticRings',
    'BalabanJ', 'PEOE_VSA6', 'PEOE_VSA7', 'SMR_VSA5', 'EState_VSA4'
]

# Create the plots
plot_shap_dependence(
    feature_list=feature_list,
    X_train_selected=X_train_selected,  # Selected features array
    shap_values_array=shap_values_class_1,
    scaler=scaler,
    feature_names=selected_feature_names,  # Names of selected features
    original_feature_names=original_feature_names,  # From X_train.columns
    file_name="SHAP_Dependence_Plots_with_annotations.png"
)
```


    
![png](DIA_RDKit_Prediction_files/DIA_RDKit_Prediction_36_0.png)
    



```python

```
