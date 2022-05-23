import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import biosppy.signals.eeg as bse
import biosppy.signals.emg as bsm
import biosppy.signals.tools as bst
import pyeeg
from sklearn import *


def load_df(filename, Id=True):
    data = pd.read_csv('./Desktop/AML/Projects/task4/' + filename + '.csv')
    
    if(Id == False):
        data = data.drop(columns='Unnamed: 0')
    else:
        data = data.drop(columns='Id')
        
    return data


def plot_ts(epoch, df):
    epoch, X_train = epoch, df
    row = X_train.iloc[epoch, :]
    plt.plot(range(len(row)), row)
    plt.title(label = 'Class:' + str(train_labels.iloc[epoch][0]))


def eeg_filter(signal, smp_rate):
    # from biosppy github
    # high pass filter
    b, a = bst.get_filter(ftype='butter',
                         band='highpass',
                         order=8,
                         frequency=4,
                         sampling_rate=smp_rate)
    
    aux, _ = bst._filter_signal(b, a, signal=signal, check_phase=True, axis=0)
    
    # low pass filter
    b, a = bst.get_filter(ftype='butter',
                         band='lowpass',
                         order=16,
                         frequency=40,
                         sampling_rate=smp_rate)
    
    filtered, _ = bst._filter_signal(b, a, signal=aux, check_phase=True, axis=0)
    return filtered


def emg_filter(signal, smp_rate):
    # from biosppy, changed freq from 100 to fit use case
    filtered, _, _ = bst.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='highpass',
                                      order=4,
                                      frequency=6,
                                      sampling_rate=smp_rate)
    return filtered


def eeg_features(signal, smp_rate, i):
    eeg_feat = {}
    
    # clean signal
    signal = np.array(signal).reshape(-1,1)
    eeg = eeg_filter(signal, smp_rate)

    # power band features
    power = bse.get_power_features(eeg, smp_rate, size=4, overlap=0).as_dict()
    total = power['theta'] + power['alpha_low'] + power['alpha_high'] + power['beta'] + power['gamma']
    eeg_feat.update({'theta' + str(i): power['theta'].item(),
                     'alpha_low' + str(i): power['alpha_low'].item(),
                     'alpha_high' + str(i): power['alpha_high'].item(),
                     'beta' + str(i): power['beta'].item(),
                     'gamma' +  str(i): power['gamma'].item(),
                     'total_pow' + str(i): total.item()
                     })
    
    # statistical features
    stats = bst.signal_stats(eeg).as_dict() 
    eeg_feat.update({'max' + str(i): stats['max'],
                     'mean' + str(i): stats['mean'],
                     'median' + str(i): stats['median'],
                     'abs_dev' + str(i): stats['abs_dev'],
                     'std_dev' + str(i): stats['std_dev'],
                     'var' + str(i): stats['var'],
                     'skewness' +  str(i): stats['skewness'].item(),
                     'kurtosis' + str(i): stats['kurtosis'].item()
                     })
    
    # pyeeg features
    eeg_l = list(eeg.ravel())
    diff = list(np.diff(eeg_l))
    hjorth_mobil, hjorth_complex = pyeeg.hjorth(eeg_l, diff)
    eeg_feat.update({'petr_fract_dim' + str(i): pyeeg.pfd(eeg_l, diff),
                     'hig_fract_dim' + str(i): pyeeg.hfd(eeg_l, Kmax=8),
                     'hjorth_mobil' + str(i): hjorth_mobil,
                     'hjorth_complex' + str(i): hjorth_complex,
                     'detrend_fluct' + str(i): pyeeg.dfa(eeg_l),
                     'hurst' + str(i): pyeeg.hurst(eeg_l)
                     })      
    return eeg_feat


def emg_features(signal, smp_rate, i):
    emg_feat = {}
    
    # clean signal
    signal = np.array(signal)
    emg = emg_filter(signal, smp_rate)

    # features from TS, freq and stats domains

    stats = bst.signal_stats(emg).as_dict()
    int_abs_val = np.sum(np.abs(emg)) 
    sq_integral = np.sum(emg ** 2)
    N, diff = len(emg), np.diff(emg)
    
    emg_feat.update({'max' + str(i): stats['max'],
                     'mean' + str(i): stats['mean'],
                     'median' + str(i): stats['median'],
                     'std_dev' + str(i): stats['std_dev'],
                     'var' + str(i): stats['var'],
                     'skewness' +  str(i): stats['skewness'],
                     'kurtosis' + str(i): stats['kurtosis'],
                     'int_abs_val' + str(i): int_abs_val,
                     'mean_abs_val' + str(i): (int_abs_val/N),
                     'sq_integral' + str(i): sq_integral,
                     'root_mean_sq' + str(i): np.sqrt(sq_integral/N),
                     'waveform_len' + str(i): np.sum(np.abs(diff)),
                     'mean_amp_chg' + str(i): np.mean(np.abs(diff)),
                     'zero_crosses' + str(i): len(nk.signal_zerocrossings(emg)),
                     'samp_entropy' + str(i): nk.entropy_sample(emg, delay=smp_rate),
                     'temp_4th_ord' + str(i): np.mean(emg ** 4),
                     })
    return emg_feat

### Load data ###
scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
smp_rate = 128

train_eeg1, test_eeg1 = load_df('train_eeg1'), load_df('test_eeg1')
train_eeg2, test_eeg2 = load_df('train_eeg2'), load_df('test_eeg2')
train_emg, test_emg = load_df('train_emg'), load_df('test_emg')
train_labels = load_df('train_labels')

### Data exploration ###
print('Class 1:', np.sum(train_labels == 1)[0])
print('Class 2:', np.sum(train_labels == 2)[0])
print('Class 3:', np.sum(train_labels == 3)[0])
plot_ts(1000, train_emg)

### Extract features ###
feat_train, feat_test = {}, {}
for index in range(64800):
    
    eeg1 = train_eeg1.iloc[index]
    eeg2 = train_eeg2.iloc[index]
    emg = train_emg.iloc[index]
    
    feat1 = eeg_features(eeg1, smp_rate, i=1)
    feat2 = eeg_features(eeg2, smp_rate, i=2)
    feat3 = emg_features(emg, smp_rate, i=3)
    feat = {**feat1, **feat2, **feat3}
    
    feat_train.update({index: feat})
    print('Done features for training')
    
    if (index < 43200):
        
        eeg1_test = test_eeg1.iloc[index]
        eeg2_test = test_eeg2.iloc[index]
        emg_test = test_emg.iloc[index]
        
        feat1_test = eeg_features(eeg1_test, smp_rate, i=1)
        feat2_test = eeg_features(eeg2_test, smp_rate, i=2)
        feat3_test = emg_features(emg_test, smp_rate, i=3)
        feat_t = {**feat1_test, **feat2_test, **feat3_test}
        
        feat_test.update({index: feat_t})
        print('Done features for testing')
    
    else:
        pass
    
    print('Completed iteration', index, '\n')
     
features_train = pd.DataFrame.from_dict(feat_train, orient='index')
features_test = pd.DataFrame.from_dict(feat_test, orient='index')

#features_train.to_csv('./Desktop/AML/Projects/task4/features_train.csv')
#features_test.to_csv('./Desktop/AML/Projects/task4/features_test.csv')
# Reload features from repo
features_train = load_df('features_train', False)
features_test = load_df('features_test', False)

### Model evaluation ###
X_train = features_train
X_test = features_test
y_train = np.array(train_labels).ravel()

# Scaling
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Param tuning
param_grid = {'C':list(np.logspace(-3,2,6)), 'gamma':list(np.logspace(-3,1,5))}
model = svm.SVC(kernel='rbf', class_weight='balanced', cache_size=500)
cv = model_selection.KFold(n_splits=3, shuffle=False)

grid_search = model_selection.GridSearchCV(model, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)
scores = grid_search.cv_results_['mean_test_score']
scores_std = grid_search.cv_results_['std_test_score']

### Final model ###
clf_svc = svm.SVC(C=0.01, kernel='rbf', gamma='scale', class_weight='balanced')

# Single CV Evaluation
cv = model_selection.KFold(n_splits=3, shuffle=False)
cv_score = model_selection.cross_validate(clf_svc,
                                          X_train,
                                          y_train,
                                          scoring=scorer,
                                          cv=cv,
                                          return_train_score=True,
                                          n_jobs=-1)

print('Train Accuracy: %0.3f (+/- %0.2f)' %(cv_score['train_score'].mean(), 
                                            cv_score['train_score'].std()*2))
print('Validation Accuracy: %0.3f (+/- %0.2f)' %(cv_score['test_score'].mean(), 
                                                 cv_score['test_score'].std()*2))

#C=0.001
# train [0.91725055, 0.89642583, 0.91489558]
# test [0.86635465, 0.90158838, 0.86982879]
# Train Accuracy: 0.91 (+/- 0.02)
# Validation Accuracy: 0.88 (+/- 0.03)
#C=0.01 
# train [0.94796875, 0.92235411, 0.93083258]
# test [0.90390092, 0.91581928, 0.87980422]
# Train Accuracy: 0.93 (+/- 0.02)
# Validation Accuracy: 0.90 (+/- 0.03)
#C=0.1
# train [0.96317333, 0.93817534, 0.94169168]
# test [0.90942344, 0.87356802, 0.89493946]
# Train Accuracy: 0.95 (+/- 0.02)
# Validation Accuracy: 0.89 (+/- 0.03)
#C=1.0
# train [0.96943474, 0.94592359, 0.95067733]
# test [0.90326599, 0.85837234, 0.92141247]
# Train Accuracy: 0.96 (+/- 0.02)
# Validation Accuracy: 0.89 (+/- 0.05)

# Training and prediction
clf_svc.fit(X_train, y_train)
y_pred = clf_svc.predict(X_test)

y_pred = pd.DataFrame({'Id':range(len(y_pred)), 'y':y_pred})
y_pred.to_csv('./Desktop/AML/Projects/task4/y_pred.csv', index=False)
