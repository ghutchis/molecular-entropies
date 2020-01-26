
# inspired by https://iwatobipen.wordpress.com/2016/12/13/build-regression-model-in-keras/

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras import regularizers

def getFpArrSmiles( smiles, radius=2, nBits=1024 ):
    X = []
    for line in smiles:
        try:
            m = Chem.MolFromSmiles(line)
            fp = AllChem.GetMorganFingerprintAsBitVect( m, 2, nBits=nBits )
        except Boost.Python.ArgumentError:
            continue # mis-formed
        arr = np.zeros( (1,) )
        DataStructs.ConvertToNumpyArray( fp, arr )
        X.append( arr )
    return X

def getFpArr( fps ):
    X = []
    for item in fps:
        bv = DataStructs.ExplicitBitVect(4096)
        DataStructs.ExplicitBitVect.FromBase64(bv, item)
        arr = np.zeros( (1,) )
        DataStructs.ConvertToNumpyArray( bv, arr )
        X.append(arr)
    return X

def base_model(nBits = 4096, hidden=int(sys.argv[1]), layers=int(sys.argv[2]), drop=float(sys.argv[3])):
    elastic=regularizers.l1_l2(l1=0.05, l2=0.05)
    model = Sequential()
    model.add( Dense( input_dim=nBits, units = hidden, kernel_regularizer=elastic ) )
    model.add( Activation( "relu" ) )

    inner = hidden # how many for a given inner layer
    for layer in range(layers):
        model.add(Dropout(drop))
        inner = inner / 2
        model.add( Dense( units=hidden, kernel_regularizer=elastic ) )
        model.add( Activation( "relu" ) )
    model.add( Dense( 1, activation="linear" ) )
    #model.add( Activation( 'relu' ) )
    model.compile( loss="mean_absolute_error",  optimizer="adam" )
    return model


if __name__ == '__main__':

    df = pd.read_csv(sys.argv[5])
    print( "Size: {}".format(len(df.index)) )

    nBits = 4096
    #X = getFpArrSmiles(df.SMILES,  nBits=nBits)
    X = getFpArr(df.ECFP6)
    Y = df.ConfEntropy

    trainx, testx, trainy, testy = train_test_split( X, Y, test_size=0.2, random_state=0 )
    trainx, testx, trainy, testy = np.asarray( trainx ), np.asarray( testx ), np.asarray( trainy ), np.asarray( testy )
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    estimator = KerasRegressor( build_fn = base_model,
                                epochs=int(sys.argv[4]),
                                shuffle=True,
                                verbose=0,
                                 )
    history = estimator.fit( trainx, trainy, validation_data=(testx, testy), callbacks=[es] )
    best_y = estimator.predict( trainx )
    pred_y = estimator.predict( testx )
    r2 = r2_score( testy, pred_y )
    mae = mean_absolute_error( testy, pred_y )
    print( "KERAS: R^2 : {0:f}, MAE : {1:f} {2:f}".format( r2, mae, mean_absolute_error(trainy, best_y) ) )
