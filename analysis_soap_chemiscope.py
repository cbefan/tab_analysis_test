import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import chemiscope
from collections import deque
from scipy.spatial.transform import Rotation as R
import ase.io
import signac
from sklearn.neighbors import KernelDensity


directory = '/mnt/researchdrive/charles/chark/lammps/oplsaa/tab/tab_425'
write_directory = '/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/chemiscope'
project = signac.get_project(directory)

n_samples = 3
cutoff_radius = 7.0
bandwidth = 0.2
resolution = 1000


with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/layering.pkl','rb') as f:
    layering_dic = pickle.load(f)

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/Rgs.pkl','rb') as f:
    Rgs_dic = pickle.load(f)

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/densities.pkl','rb') as f:
    densities_dic = pickle.load(f)

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/soap_features.pkl','rb') as f:
    features_dic = pickle.load(f)

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/soap_frames.pkl','rb') as f:
    frames_dic = pickle.load(f)

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/soap_idx.pkl','rb') as f:
    idxs_dic = pickle.load(f)
    
features = []
layers = []
Rgs = []
densities = []
frames = []
traj = []
pcovs = []
peaks = []
idxs = []
for job in layering_dic:
    features.extend(features_dic[job])
    layers.extend(layering_dic[job])
    frames.extend(frames_dic[job])
    #idxs.append(idxs_dic[job])
    #Rgs.extend(Rgs_dic[job])

    job_features = features_dic[job]
    counter = -1
    magnitude = layering_dic[job][1]
    for frame,idxs in zip(frames_dic[job],idxs_dic[job]):
        N_mol = int(len(frame)/66)
        #sample_indices = np.random.choice(np.cumsum(np.ones(N_mol))-1,size=n_samples,replace=False).astype(int)
        #distances = frame.get_all_distances(mic=True,vector=False)
        for i in idxs:
            counter += 1
            Rgs.append(Rgs_dic[job][counter])
            densities.append(densities_dic[job][counter])
            #mol_indices = [i,i+N_mol,i+2*N_mol,i+3*N_mol]
            mol_indices = np.array(range(66*i,66*(i+1)))
            #distances = frame.get_distances(mol_indices,np.array(range(len(frame))),mic=True,vector=False)
            distances = []
            for mol_index in mol_indices:
                distances.append(frame.get_distances(mol_index,np.array(range(len(frame))),mic=True,vector=False))
            distances = np.array(distances)
            mol_in_cutoff = distances[0,:]<-1
            for mol_index in range(len(mol_indices)):
                #mol_in_cutoff += distances[mol_index,:]<cutoff_radius
                mol_in_cutoff += distances[mol_index,:]<cutoff_radius
            
            indices = np.where(mol_in_cutoff>=1)[0]
            indices[indices==i] = indices[0]
            mol_frame = frame[indices]
            mol_frame.positions -= mol_frame.get_center_of_mass()#frame.positions[i]
            mol_frame.positions += np.diag(mol_frame.cell)/2
            mol_frame.wrap()
            boolean_indices = np.full(len(frame),True,dtype=bool)
            boolean_indices[mol_indices] = False
            boolean_indices = boolean_indices[indices]
            #mol_frame.numbers[boolean_indices] = 1*np.ones(len(mol_frame[boolean_indices]))

            traj.append(mol_frame)
            mol_feature = job_features[counter]
            pcovs.append(mol_feature)
            peaks.append(magnitude)
        #idx += N_mol
    #break
pcovs = np.array(pcovs)

properties = {      # the data
    "PCovR": {
        "target": "structure",
        "values": pcovs,
        "description": "",
    },
    "Peak Magnitude": {
        "target": "structure",
        "values": peaks,
        "description": ""
    },
    "Radius of Gyration": {
        "target": "structure",
        "values": Rgs,
        "description": ""
    },
    "Density": {
        "target": "structure",
        "values": densities,
        "description": ""
    },
}

environments = [   
    (i, j, cutoff_radius) for i in range(len(traj)) for j in range(len(traj[i]))
]


chemiscope.write_input(
    path=f"{write_directory}/soap_chemiscope.json",
    frames=traj,
    properties=properties,
    environments=environments,
)


kde = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(pcovs[:,:2])
x_maximum = np.max(pcovs[:,0])
x_minimum = np.min(pcovs[:,0])
y_maximum = np.max(pcovs[:,1])
y_minimum = np.min(pcovs[:,1])
grid = np.meshgrid(*[np.linspace(x_minimum,x_maximum,resolution),np.linspace(y_minimum,y_maximum,resolution)])

loglikelihoods = kde.score_samples(np.reshape(grid,(2,-1)).T)
loglikelihoods = np.reshape(loglikelihoods,(resolution,resolution))

plt.imshow(np.exp(loglikelihoods),origin='lower',extent=[x_minimum,x_maximum,y_maximum,y_minimum],aspect='auto')
#plt.title(f'Ratio = {r}')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig(f'/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/kde/soap_kde.png')

print('goodbye')
