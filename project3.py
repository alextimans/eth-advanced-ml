import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import tsfresh, tsfel  
from sklearn import *
from collections import OrderedDict

def ecg_process(ecg, smp_rate, visualize=False):
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=smp_rate)
    
    # R peaks + correct peaks
    inst_rpeaks, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=smp_rate, correct_artifacts=True)
    rpeaks.update({'ECG_R_Peaks':list(OrderedDict.fromkeys(rpeaks['ECG_R_Peaks']))})
    
    # Signal rate + signal quality
    rate = nk.ecg_rate(rpeaks, sampling_rate=smp_rate, desired_length=len(ecg_clean))
    quality = nk.ecg_quality(ecg_clean, rpeaks=None, sampling_rate=smp_rate)
    
    # P, Q, S, T peaks, P onset, T offset
    inst_peaks, peaks = nk.ecg_delineate(ecg_clean, rpeaks=rpeaks, sampling_rate=smp_rate, method='peak', show=visualize, show_type='all')

    # Cardiac phase 0/1 + perc completion
    cardiac_phase = nk.ecg_phase(ecg_clean, rpeaks=rpeaks, delineate_info=peaks, sampling_rate=smp_rate)
    
    # group information
    peak_info = {**rpeaks, **peaks}
    signal = pd.DataFrame({"ECG_Raw":ecg, "ECG_Clean":ecg_clean, "ECG_Rate":rate, "ECG_Quality":quality})
    signal = pd.concat([signal, inst_rpeaks, inst_peaks, cardiac_phase], axis=1)
    
    if visualize:
        # raw vs. clean ecg 
        plt.plot(range(0, len(ecg)), ecg)
        plt.plot(range(0, len(ecg_clean)), ecg_clean)
        # R peaks
        nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_clean) 
        # summary plot
        nk.ecg_plot(signal, sampling_rate=smp_rate)
    
    return signal, peak_info


def replace_peak_nan(peak_list):
    #replace nans with corresp. median distance between peaks  
    peak_list = np.array(peak_list)
    diff_list = np.diff(peak_list)
    
    #case if nans cause diff_list to be empty
    if(np.all(np.isnan(diff_list))):
        ind_non_nan = np.where(~np.isnan(peak_list))[0]
        diff_list = []           
        for i in range(0, len(ind_non_nan)-1):
            ind, next_ind = ind_non_nan[i], ind_non_nan[i+1]
            diff = (peak_list[next_ind] - peak_list[ind])/(next_ind - ind)
            diff_list.append(diff)
        insert_val = int(np.min(diff_list))
    else:
        insert_val = int(np.nanmedian(diff_list))
   
    ind_nan = np.where(np.isnan(peak_list))[0]
    
    #case if nan at first index
    if (0 in ind_nan):
        next_non_nan = next(val for val in peak_list if ~np.isnan(val))
        ind_next = np.where(peak_list == next_non_nan)[0][0]
        peak_list[0] = next_non_nan - ind_next * insert_val
        ind_nan = np.delete(ind_nan, 0)

    for ind in ind_nan:
        peak_list[ind] = peak_list[ind-1] + insert_val
    
    #case if calc peak out of bounds        
    if (np.any(peak_list > len(ecg))):
        ind_outofbound = np.where(peak_list > len(ecg))[0]
        for ind in ind_outofbound:
            peak_list[ind] = len(ecg)-1
        
    return list(peak_list.astype(int))


def extract_features(signal, peak_info, smp_rate, visualize=False):
    
    ecg_clean = np.array(signal['ECG_Clean'])
    
    # hrv features (mod)
    hrv = nk.hrv(peak_info, sampling_rate=smp_rate, show=visualize).squeeze()
    capen = pd.Series([nk.complexity_capen(ecg_clean)], index=['HRV_CApEn'])
    hrv.drop(['HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF'
              , 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_ApEn', 'HRV_SampEn'], inplace=True)
    hrv = hrv.append(capen)

    #replace peak_info dictionary nans with values
    peak_info.update({'ECG_P_Peaks':replace_peak_nan(peak_info['ECG_P_Peaks'])})
    peak_info.update({'ECG_Q_Peaks':replace_peak_nan(peak_info['ECG_Q_Peaks'])})
    peak_info.update({'ECG_R_Peaks':replace_peak_nan(peak_info['ECG_R_Peaks'])})
    peak_info.update({'ECG_S_Peaks':replace_peak_nan(peak_info['ECG_S_Peaks'])})
    peak_info.update({'ECG_T_Peaks':replace_peak_nan(peak_info['ECG_T_Peaks'])})
    peak_info.update({'ECG_P_Onsets':replace_peak_nan(peak_info['ECG_P_Onsets'])})
    peak_info.update({'ECG_T_Offsets':replace_peak_nan(peak_info['ECG_T_Offsets'])})    

    manual_feat = {}
    
    #min, max, mean, std, interqt of heart rate + max, mean, std, interqt diff in heart rate
    rate = np.array(signal['ECG_Rate'])
    diff_rate = np.diff(rate)
    manual_feat.update({'MF_MinRate':np.min(rate), 'MF_MaxRate':np.max(rate)
                    , 'MF_MeanRate':np.mean(rate), 'MF_StdRate':np.std(rate)
                    , 'MF_InterQtRate':(np.quantile(rate, .75) - np.quantile(rate, .25))
                    , 'MF_MaxDiffRate':np.max(diff_rate), 'MF_MeanDiffRate':np.mean(diff_rate)
                    , 'MF_StdDiffRate':np.std(diff_rate)
                    , 'MF_InterQtDiffRate':(np.quantile(diff_rate, .75) - np.quantile(diff_rate, .25))})
    
    #min, max, mean, std, interqt of quality + max, mean, std, interqt diff in quality
    quality = np.array(signal['ECG_Quality'])
    diff_quality = np.diff(quality)
    manual_feat.update({'MF_MinQuality':np.min(quality), 'MF_MaxQuality':np.max(quality)
                    , 'MF_MeanQuality':np.mean(quality), 'MF_StdQuality':np.std(quality)
                    , 'MF_InterQtQuality':(np.quantile(quality, .75) - np.quantile(quality, .25))
                    , 'MF_MaxDiffQuality':np.max(diff_quality), 'MF_MeanDiffQuality':np.mean(diff_quality)
                    , 'MF_StdDiffQuality':np.std(diff_quality)
                    , 'MF_InterQtDiffQuality':(np.quantile(diff_quality, .75) - np.quantile(diff_quality, .25))})
    
    #min, max, mean, std, interqt values of P, Q, R, S, T peaks + mean, std of ecg signal + count zero crossings
    count_zerocross = nk.signal_zerocrossings(ecg_clean).shape[0]
    peak_P = ecg_clean[peak_info['ECG_P_Peaks']]
    peak_Q = ecg_clean[peak_info['ECG_Q_Peaks']]
    peak_R = ecg_clean[peak_info['ECG_R_Peaks']]
    peak_S = ecg_clean[peak_info['ECG_S_Peaks']]
    peak_T = ecg_clean[peak_info['ECG_T_Peaks']]
    
    # not actually amplitudes but peak values + mean signal as baseline (close to zero)
    manual_feat.update({'MF_MinAmpP':np.min(peak_P), 'MF_MaxAmpP':np.max(peak_P)
                    , 'MF_MeanAmpP':np.mean(peak_P), 'MF_StdAmpP':np.std(peak_P)
                    , 'MF_InterQtAmpP':(np.quantile(peak_P, .75) - np.quantile(peak_P, .25))
                    , 'MF_MinAmpQ':np.min(peak_Q), 'MF_MaxAmpQ':np.max(peak_Q)
                    , 'MF_MeanAmpQ':np.mean(peak_Q), 'MF_StdAmpQ':np.std(peak_Q)
                    , 'MF_InterQtAmpQ':(np.quantile(peak_Q, .75) - np.quantile(peak_Q, .25))
                    , 'MF_MinAmpR':np.min(peak_R), 'MF_MaxAmpR':np.max(peak_R)
                    , 'MF_MeanAmpR':np.mean(peak_R), 'MF_StdAmpR':np.std(peak_R)
                    , 'MF_InterQtAmpR':(np.quantile(peak_R, .75) - np.quantile(peak_R, .25))
                    , 'MF_MinAmpS':np.min(peak_S), 'MF_MaxAmpS':np.max(peak_S)
                    , 'MF_MeanAmpS':np.mean(peak_S), 'MF_StdAmpS':np.std(peak_S)
                    , 'MF_InterQtAmpS':(np.quantile(peak_S, .75) - np.quantile(peak_S, .25))
                    , 'MF_MinAmpT':np.min(peak_T), 'MF_MaxAmpT':np.max(peak_T)
                    , 'MF_MeanAmpT':np.mean(peak_T), 'MF_StdAmpT':np.std(peak_T)
                    , 'MF_InterQtAmpT':(np.quantile(peak_T, .75) - np.quantile(peak_T, .25))
                    , 'MF_MeanECGClean':np.mean(ecg_clean), 'MF_StdECGClean':np.std(ecg_clean)
                    , 'MF_ZeroCrossByLength':(count_zerocross/len(ecg_clean))*100})
    
    #min, max, mean, std, interqt duration of QRS, P onset to T offset, P onset to Q, S to T offset
    qrs = np.array(peak_info['ECG_S_Peaks']) - np.array(peak_info['ECG_Q_Peaks'])
    pt_interval = np.array(peak_info['ECG_T_Offsets']) - np.array(peak_info['ECG_P_Onsets'])
    pq_interval = np.array(peak_info['ECG_Q_Peaks']) - np.array(peak_info['ECG_P_Onsets'])
    st_interval = np.array(peak_info['ECG_T_Offsets']) - np.array(peak_info['ECG_S_Peaks'])
    
    manual_feat.update({'MF_MinQRS':np.min(qrs), 'MF_MaxQRS':np.max(qrs)
                    , 'MF_MeanQRS':np.mean(qrs), 'MF_StdQRS':np.std(qrs)
                    , 'MF_InterQtQRS':(np.quantile(qrs, .75) - np.quantile(qrs, .25))
                    , 'MF_MinPT':np.min(pt_interval), 'MF_MaxPT':np.max(pt_interval)
                    , 'MF_MeanPT':np.mean(pt_interval), 'MF_StdPT':np.std(pt_interval)
                    , 'MF_InterQtPT':(np.quantile(pt_interval, .75) - np.quantile(pt_interval, .25))
                    , 'MF_MinPQ':np.min(pq_interval), 'MF_MaxPQ':np.max(pq_interval)
                    , 'MF_MeanPQ':np.mean(pq_interval), 'MF_StdPQ':np.std(pq_interval)
                    , 'MF_InterQtPQ':(np.quantile(pq_interval, .75) - np.quantile(pq_interval, .25))
                    , 'MF_MinST':np.min(st_interval), 'MF_MaxST':np.max(st_interval)
                    , 'MF_MeanST':np.mean(st_interval), 'MF_StdST':np.std(st_interval)
                    , 'MF_InterQtST':(np.quantile(st_interval, .75) - np.quantile(st_interval, .25))})
    
    manual_feat.update(hrv.to_dict())

    return manual_feat

### Load data
X_train_load = pd.read_csv('./Desktop/AML/Projects/task3/X_train.csv').drop(columns='id')
y_train_load = np.array(pd.read_csv('./Desktop/AML/Projects/task3/y_train.csv').drop(columns='id'))
X_test_load = pd.read_csv('./Desktop/AML/Projects/task3/X_test.csv').drop(columns='id')
scorer = metrics.make_scorer(metrics.f1_score, average='micro')
smp_rate = 300

### Store data locally
X_train = X_train_load
y_train = y_train_load
X_test = X_test_load

### Data exploration
pd.DataFrame(X_train).isna().sum().sum()
pd.DataFrame(X_test).isna().sum().sum()
plt.hist(y_train, edgecolor='black', bins=10)
row = 76
X_train_row = X_train.iloc[row,:].dropna()
plt.plot(range(0, len(X_train_row)), X_train_row)
plt.title(label='Class: ' + str(y_train[row]))

### ECG processing
dict_feat = {}
for index in range(60, 80):
    ecg = np.array(X_train.iloc[index,:].dropna().reset_index(drop=True))
    signal, peak_info = ecg_process(ecg, smp_rate)
    feat = extract_features(signal, peak_info, smp_rate)
    dict_feat.update({index:feat})
    print('Features calculated for sample', index)
      
features = pd.DataFrame.from_dict(dict_feat, orient='index')

### Testing
beats = nk.ecg_segment(signal['ECG_Clean'], rpeaks=None, sampling_rate=smp_rate, show=True)
se = nk.entropy_shannon(ecg_clean)
cp = nk.signal_changepoints(signal['ECG_Clean'], show=True)
f, t, stft = nk.signal_timefrequency(signal['ECG_Clean'], sampling_rate=smp_rate, method='stft', show=False)

### Params tuning
param_grid = {'C':[1e-2, 1e-1, 1e0, 1e1, 1e2], 'gamma':[1e-3, 1e-2, 1e-1, 1e0, 1e1]}
model = svm.SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
grid_search = model_selection.GridSearchCV(model, param_grid, scoring=scorer, cv=8, n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())

### Final model
final_model = svm.SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
cv_score = model_selection.cross_val_score(final_model, X_train, y_train.ravel(), cv=8, scoring=scorer, n_jobs=-1)
print('CV Score:', cv_score.mean())

final_model.fit(X_train, y_train.ravel())
y_pred = final_model.predict(X_test)
y_pred = pd.DataFrame({'id': range(0, len(y_pred)), 'y': y_pred})
y_pred.to_csv('./Desktop/AML/Projects/task3/y_pred.csv', index=False)
