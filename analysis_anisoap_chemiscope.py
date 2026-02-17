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

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/Rg2s.pkl','rb') as f:
    Rg2s_dic = pickle.load(f)

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/features.pkl','rb') as f:
    features_dic = pickle.load(f)

with open('/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles/asoap_frames.pkl','rb') as f:
    frames_dic = pickle.load(f)

    
features = []
layers = []
Rg2s = []
frames = []
traj = []
pcovs = []
peaks = []
for job in layering_dic:
    features.extend(features_dic[job])
    layers.extend(layering_dic[job])
    frames.extend(frames_dic[job])
    #Rg2s.extend(Rg2s_dic[job])

    job_features = features_dic[job]
    idx = 0
    magnitude = layering_dic[job][1]
    for frame in frames_dic[job]:
        N_mol = int(len(frame)/4)
        sample_indices = np.random.choice(np.cumsum(np.ones(N_mol))-1,size=n_samples,replace=False).astype(int)
        distances = frame.get_all_distances(mic=True,vector=False)
        for i in sample_indices:
            Rg2s.append(Rg2s_dic[job][i+idx])
            mol_indices = [i,i+N_mol,i+2*N_mol,i+3*N_mol]
            mol_in_cutoff = distances[0,:]<-1
            for mol_index in mol_indices:
                mol_in_cutoff += distances[mol_index,:]<cutoff_radius
            frame.positions -= frame.positions[i]
            frame.positions += np.diag(frame.cell)/2
            frame.wrap()
            frame.numbers = 1*np.ones(len(frame))
            frame.numbers[mol_indices] = 3*np.ones(len(frame[mol_indices]))
            frame.numbers[i] = 4
            indices = np.where(mol_in_cutoff>=1)[0]
            indices[indices==i] = indices[0]
            indices[0] = i
            mol_frame = frame[indices]
            #frame_dic = mol_frame.todict()
            #diameters = frame_dic['c_diameter[1]']
            #quats = frame_dic['c_q']
            #rotations = R.from_quat(quats,scalar_first=True)
            #vecs = np.zeros((quats.shape[0],3))
            ##vecs[:,2] = 1
            #vecs[diameters==2,2] = 1
            #vecs[diameters!=2,0] = 1
            #vecs = rotations.apply(vecs)

            traj.append(mol_frame)
            #mol_feature = job_features[i+idx:(i+1)+idx]
            mol_feature = job_features[i+idx]
            #print(mol_feature)
            pcovs.append(mol_feature)
            peaks.append(magnitude)

            #print(job_features.shape)
            #print(mol_feature.shape)
            #print(i+idx,(i+1)+idx)
        idx += N_mol
    #break
pcovs = np.array(pcovs)
print(pcovs.shape)

properties = {      # the data
    "PCovR": {
        "target": "structure",
        "values": pcovs,
        "description": "PCovR of per-atom representation of the structures",
    },
    "Peak Magnitude": {
        "target": "structure",
        "values": peaks,
        "description": ""
    },
    "Radius of Gyration": {
        "target": "structure",
        "values": Rg2s,
        "description": ""
    },
}

environments = [   
    (i, j, cutoff_radius) for i in range(len(traj)) for j in range(len(traj[i]))
]

shape_list = []
for frame in traj:
    #frame = traj[i]
    dic = frame.todict()
    #print(dic)
    diam1 = dic['c_diameter[1]']
    diam2 = dic['c_diameter[2]']
    diam3 = dic['c_diameter[3]']
    quats = dic['c_q']
    for j in range(len(frame)):
        semiaxes = [diam1[j],diam2[j],diam3[j]]
        orientation = quats[j].tolist()
        #orientation = np.roll(orientation,-1)
        d = deque(orientation)
        d.rotate(-1)
        orientation = list(d)
        shape = {'semiaxes':semiaxes,'orientation':orientation}
        shape_list.append(shape)
shapes = {
    "center": {
        "kind": "ellipsoid",
        "parameters": {
            "atom": shape_list
        }
    }
}

chemiscope.write_input(
    path=f"{write_directory}/chemiscope.json",
    frames=traj,
    properties=properties,
    environments=environments,
    shapes=shapes
)


kde = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(pcovs[:,:2])
x_maximum = np.max(pcovs[:,0])
x_minimum = np.min(pcovs[:,0])
y_maximum = np.max(pcovs[:,1])
y_minimum = np.min(pcovs[:,1])
grid = np.meshgrid(*[np.linspace(x_minimum,x_maximum,resolution),np.linspace(y_minimum,y_maximum,resolution)])

loglikelihoods = kde.score_samples(np.reshape(grid,(2,-1)).T)
loglikelihoods = np.reshape(loglikelihoods,(resolution,resolution))

plt.imshow(np.exp(loglikelihoods),origin='lower',extent=[x_minimum,x_maximum,y_maximum,y_minimum])
#plt.title(f'Ratio = {r}')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig(f'/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/kde/kde.png')

print('goodbye')
