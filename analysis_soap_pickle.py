import numpy as np
import matplotlib.pyplot as plt
import signac
import pandas as pd
from skmatter.preprocessing import StandardFlexibleScaler
import ase.io
from featomic import SoapPowerSpectrum
import pickle
from analysis_anisoap_helper import *
import dynasor
from ase import Atoms
import os
from ase.geometry import wrap_positions
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from metatensor import Labels
dynasor.logging_tools.set_logging_level(30)

subsample = 100
begin = 0
end = int(1.1e6)
cutoff_radius = 7.0#15.0
scrungus = 1#0
n_components = 5
lmax = 3#6#4#5
nmax = 2#8#6#3
gaussian_width = 0.2#3.0#1.5
alpha = 0.5
n_components = 5
n_samples = 3


directory = '/mnt/researchdrive/charles/chark/lammps/oplsaa/tab/tab_425'
write_directory = '/mnt/researchdrive/charles/chark/tab_data/anisoap/pcovr/pickles'
project = signac.get_project(directory)
to_gcm3 = (40*12.011+26*1.008)*((1e8)**3)/(6.022e23)
    

print('hello')


# CALCULATE SOAPS
def saponify_frame(frame,calculator,selected_samples):
    feature_vector = calculator.compute(frame,selected_samples=selected_samples)
    feature_vector = feature_vector.keys_to_samples("center_type")
    feature_vector = feature_vector.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    feature_vector = feature_vector.block().values
    return feature_vector

hypers = {
    "cutoff": {
        "radius": cutoff_radius,
        "smoothing": {
            #"type": "Step"
            "type": "ShiftedCosine",
            "width": cutoff_radius/5
        }
    },
    "density": {
        "type": "Gaussian",
        "width": gaussian_width
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": lmax,
        "radial": {
            "type": "Gto",
            "max_radial": nmax
        }
    }
}
calculator = SoapPowerSpectrum(**hypers)

soap_dic = {}
idx_dic = {}
for job in project:
    T = job.sp.T
    t_frames = np.arange(begin,end,job.sp.frame_step*subsample,dtype=int)
    frames = []
    idxs = []
    selection = []
    frame_idx = 0
    for t in t_frames:
        frame = ase.io.read(job.fn(f'trajectory_prod/full_{T}_{t}.xyz'))
        N_mol = int(len(frame)/66)
        sample_indices = np.random.choice(np.cumsum(np.ones(N_mol))-1,size=n_samples,replace=False).astype(int)
        for i in sample_indices:
            mol_indices = np.array(range(66*i,66*(i+1)))
            distances = []
            for mol_index in mol_indices:
                #distances.append(frame.get_distances(mol_index,np.array(range(len(frame))),mic=True,vector=False))
                selection.append([frame_idx,mol_index])
                assert np.sum(frame[mol_indices].numbers) == 40*6 + 26*1
                #print(frame[mol_indices].numbers);exit()
            #distances = np.array(distances)
            #mol_in_cutoff = distances[0,:]<-1
            #for mol_index in range(len(mol_indices)):
            #    #mol_in_cutoff += distances[mol_index,:]<cutoff_radius
            #    mol_in_cutoff += distances[mol_index,:]<cutoff_radius
            #indices = np.where(mol_in_cutoff>=1)[0]
            #mol_frame = frame[indices]
        frames.append(frame)
        idxs.append(sample_indices)
            
        #sub_idx = np.random.choice(np.array(range(len(frame))),size=2000,replace=False)
        #idxs.append(sub_idx)
        #frames.append(frame[sub_idx])
        frame_idx += 1

    selection = np.array(selection)
    #print(selection.shape)
    selected_samples = Labels(
        names=['system','atom'],
        values=selection
    )
    #print(selection.shape)
    soap = saponify_frame(frames,calculator,selected_samples)
    #print(soap.shape)
    test = np.sum(soap[:66,:])
    #print(np.array(soap).shape)
    #print(np.array(idxs).shape)
    soap = np.reshape(soap,(round(selection.shape[0]/66),soap.shape[1],-1))
    #print(soap.shape)
    soap = np.hstack((np.average(soap[:,:,:40],axis=2),np.average(soap[:,:,40:],axis=2)))
    #soap = np.average(soap,axis=2)
    #print(soap.shape)
    #assert np.sum(soap[0]) == test
    
    soap_dic[job] = soap
    idx_dic[job] = idxs
    #exit()

with open(f'{write_directory}/soap.pkl','wb') as f:
    pickle.dump(soap_dic,f)

with open(f'{write_directory}/soap_idx.pkl','wb') as f:
    pickle.dump(idx_dic,f)

print('calculated soaps')




# CALCULATE RADII OF GYRATION
def guess_bonds(atoms,scale_factor=1.2):
    cutoffs = []
    for Z in atoms.numbers:
        r = covalent_radii[Z]
        cutoffs.append(r*scale_factor)

    nl = NeighborList(cutoffs,self_interaction=False,bothways=True)
    nl.update(atoms)

    bonds = []
    adjacency = {i:set() for i in range(len(atoms))}
    for i in range(len(atoms)):
        indices,offsets = nl.get_neighbors(i)
        for j in indices:
            if i<j:
                bonds.append((i,j))
            adjacency[i].add(j)
            adjacency[j].add(i)

    visited = set()
    molecules = []

    for start in range(len(atoms)):
        if start not in visited:
            stack = [start]
            component = []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    stack.extend(adjacency[node] - visited)
            molecules.append(sorted(component))

    return molecules#,bonds

Rgs_dic = {}
frames_dic = {}
densities_dic = {}
flag = 0
for job in project:
    T = job.sp.T
    t_frames = np.arange(begin,end,job.sp.frame_step*subsample,dtype=int)
    Rgs = []
    frames = []
    densities = []
    for t in t_frames:
        full_frame = ase.io.read(job.fn(f'trajectory_prod/full_{T}_{t}.xyz'))
        frames.append(full_frame)
        #print(guess_bonds(full_frame)[0])
        #molecules = np.array(guess_bonds(full_frame))
        molecules = [np.array(range(66*i,66*(i+1))) for i in range(int(len(full_frame)/66))]
        N = len(molecules)
        vol = full_frame.get_volume()
        density = N/vol * to_gcm3
        for molecule in molecules:
            frame = full_frame[molecule]
            masses = frame.get_masses()[np.newaxis,:]
            distances = frame.get_all_distances(mic=True,vector=True)
            distances = np.diagonal(distances,offset=1)
            distances = np.cumsum(distances,axis=1)
            distances = np.hstack((np.zeros((3,1)),distances))
            com = np.sum(masses*distances,axis=1)/np.sum(masses)
            distances = distances - com[:,np.newaxis]
            Rg2 = np.sum(masses*distances**2)/np.sum(masses)
            if flag == 1:
                ase.io.write(f'test/{job}.xyz',frame)
                flag = 0
            Rgs.append(np.sqrt(Rg2))
            densities.append(density)
    frames_dic[job] = frames
    Rgs_dic[job] = Rgs
    densities_dic[job] = densities
    #exit()

#print(Rg2s_dic)
with open(f'{write_directory}/Rgs.pkl','wb') as f:
    pickle.dump(Rgs_dic,f)

with open(f'{write_directory}/densities.pkl','wb') as f:
    pickle.dump(densities_dic,f)
    
with open(f'{write_directory}/soap_frames.pkl','wb') as f:
    pickle.dump(frames_dic,f)
#exit()
print('calculated Rg2s')


    
# CALCULATE SCATTERING PEAK MAGNITUDES
flag_q_dependent = 1
# Waasmaier, D.; Kirfel, A. New Analytical Scattering-Factor Functions for Free Atoms and Ions. Acta Crystallogr A Found Crystallogr 1995, 51 (3), 416–431. https://doi.org/10.1107/S0108767394013292.
scattering_factors = {'H':[0,0,0,0,0,0,0,0,0,0,0],
                      'Li':[0.432724,0.260367,0.549257,1.042836,0.376575,7.885294,-0.336481,0.260368,0.876060,3.042539,0.001764],
                      'C':[2.657506,14.780758,1.078079,0.776775,1.490909,42.086843,-4.241070,-0.000294,0.713791,0.239535,4.297983],
                      'N':[11.893780,0.000158,3.277479,10.232723,1.858092,30.344690,0.858927,0.656065,0.912985,0.217287,-11.804902],
                      'O':[2.960427,14.182259,2.508818,5.936858,0.637853,0.112726,0.722838,34.958481,1.132756,0.390240,0.027014],
                      'F':[3.511943,10.687859,2.772244,4.380466,0.678385,0.093982,0.915159,27.255203,1.089261,0.313068,0.032557],
                      'S':[6.372157,1.514347,5.154568,22.092528,1.473732,0.061373,1.635073,55.445176,1.209372,0.6446925,0.154722]}
masses= {'H':1,'Li':3,'C':12,'N':14,'O':16,'F':19,'S':32}
z_q_norm = 0.59


frames_dic = {}
layering_dic = {}

project = signac.get_project(directory)
jobs = project.find_jobs()#{'traj':0})
z_structure_factors = []

for job in jobs:
    T = job.sp.T
    t_frames = np.arange(begin,end,job.sp.frame_step*subsample,dtype=int)
    is_looping = 0
    frames = []
    for t in t_frames[:]:
        #frame = ase.io.read(job.fn(f'trajectory_equil/full_{T}_{t}.xyz'))
        try:
            frame = ase.io.read(job.fn(f'trajectory_prod/full_{T}_{t}.xyz'))
            is_looping = 1
        except Exception as e:
            print(e)
            break
        frames.append(frame)
    frames_dic[job] = frames
    ase.io.write('scrungus2.xyz',frames)

    # z averaged
    traj = dynasor.Trajectory('scrungus2.xyz','extxyz',atomic_indices='read_from_trajectory')
    os.remove('scrungus2.xyz')
    L = traj.cell[2,2]
    k_hat = 2*np.pi/L
    #print(n_k)
    n_k = 70
    z_q_points = 2*np.pi/L*np.arange(1,n_k)
    idx = (np.abs(z_q_points-z_q_norm)).argmin()
    #z_q_point = z_q_points[idx-1]
    z_q_norms = np.array([z_q_points[idx-1],z_q_points[idx],z_q_points[idx+1],z_q_points[idx+2],z_q_points[idx+3]])
    z_q_points = np.hstack((np.zeros((z_q_norms.shape[0],2)),z_q_norms[:,np.newaxis]))
    #z_q_points = np.reshape(z_q_points,(1,3))
    sample = dynasor.compute_static_structure_factors(traj,z_q_points)
    z_structure_factor = np.zeros((5,1))#0

    z_q_norms = z_q_norms[:,np.newaxis]
    for pair in sample.pairs:
        #print(pair)
        command=f'partial_Sq=sample.Sq_{pair[0]}_{pair[1]}'
        exec(command)
        if flag_q_dependent:
            params1 = scattering_factors[pair[0]]
            params2 = scattering_factors[pair[1]]
            factor1 = params1[10]*np.ones((z_q_norms.shape[0],1))
            factor2 = params2[10]*np.ones((z_q_norms.shape[0],1))
            for i in range(5):
                factor1 += params1[2*i]*np.exp(-params1[2*i+1]*(z_q_norms/4/np.pi)**2)
                factor2 += params2[2*i]*np.exp(-params2[2*i+1]*(z_q_norms/4/np.pi)**2)
            z_structure_factor += factor1*factor2*partial_Sq
        else:
            z_structure_factor += masses[pair[0]]*masses[pair[1]]*partial_Sq

    #z_structure_factor = list(np.reshape(z_structure_factor,(5)))
    #z_structure_factors.extend([np.max(z_structure_factor)])
    #print(z_structure_factor)
    #data = np.hstack((z_q_norms,z_structure_factor)).tolist()
    data = [z_q_norms[z_structure_factor==np.max(z_structure_factor)][0],np.max(z_structure_factor)]
    layering_dic[job] = data
    #print(layering_dic)
    #exit()

#np.savetxt(f'{write_directory}/layering_dic.json',z_structure_factors)
#plt.plot(np.log(sorted(z_structure_factors)))
#plt.show()

with open(f'{write_directory}/layering.pkl','wb') as f:
    pickle.dump(layering_dic,f)

print('calculated scattering peaks')


# FIT PCOVR
soaps = []
layers = []
qnorms = []
for job in soap_dic:
    soaps.extend(soap_dic[job])
    N_job = len(soap_dic[job])
    qnorms.extend([layering_dic[job][0] for i in range(N_job)])
    layers.extend([layering_dic[job][1] for i in range(N_job)])
    #exit()

scaling = StandardFlexibleScaler(column_wise=False)
scaling.fit(pd.DataFrame(soaps))

features = scaling.transform(soaps)

pcovr = PCovR(mixing=alpha,n_components=n_components)
pcovr.fit(features,layers)
pcovr_dic = {}
for job in soap_dic:
    pcovr_dic[job] = pcovr.transform(soap_dic[job])

#latent_features = pcovr.transform(features)

with open(f'{write_directory}/soap_features.pkl','wb') as f:
    pickle.dump(pcovr_dic,f)

print('calculated pcovr')
    
    
print('goodbye')

