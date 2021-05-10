import pickle

studynames = pickle.load(open('CTscans_studynames_train', 'rb'))
volumes = pickle.load(open('CTscans_3Dvolumes_train', 'rb'))
labels = pickle.load(open('CTscans_3Dlabels_train', 'rb'))

print(volumes[0].shape) # one 3D volume