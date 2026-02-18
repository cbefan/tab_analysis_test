import numpy as np
from skmatter.decomposition import PCovC, PCovR
import matplotlib.pyplot as plt
import signac
from featomic import SoapPowerSpectrum
import pandas as pd
from skmatter.preprocessing import StandardFlexibleScaler
import ase.io
from sklearn.linear_model import LogisticRegression
import chemiscope
import os
import anisoap
from anisoap.representations.ellipsoidal_density_projection import EllipsoidalDensityProjection
from ase import Atoms
from scipy.spatial.transform import Rotation as R
import pickle
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from freud.order import Nematic
from scipy.stats import ttest_ind
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.patches as mpatches



def get_moments_of_inertia(frame: Atoms, vectors=False):
    """
    COPIED FROM ASE Source: https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_moments_of_inertia
    MODIFIED TO Include MINIMUM IMAGE CONVENTION. 
    Get the moments of inertia along the principal axes.

    The three principal moments of inertia are computed from the
    eigenvalues of the symmetric inertial tensor. Periodic boundary
    conditions are ignored. Units of the moments of inertia are
    amu*angstrom**2.
    """
    displacements = frame.get_all_distances(mic=True, vector=True)  # vectors is a nxnx3 array, it's the displacement from each point to all other points
    com = frame[0].position + displacements[0].mean(axis=0)
    # com = frame.get_center_of_mass()
    positions = frame[0].position + displacements[0] 
    positions -= com  # translate center of mass to origin
    masses = frame.get_masses()

    # Initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(frame)):
        x, y, z = positions[i]
        m = masses[i]

        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    Itensor = np.array([[I11, I12, I13],
                        [I12, I22, I23],
                        [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(Itensor)
    if vectors:
        return evals, evecs.transpose()
    else:
        return evals



    
def anisaponify_frame(frames,calculator,scrungus):
    #test = frames[0].todict();print(test['c_diameter[1]'].shape);print(test['c_diameter[1]'][:1153]);print(test['c_diameter[1]'][1152:2*1152+1]);exit()
    features = calculator.power_spectrum(frames,mean_over_samples=False)
    if scrungus == 1:   # distinguish ellipsoid species
        a = features.block(0).values.squeeze()
        b = features.block(1).values.squeeze()
        c = features.block(2).values.squeeze()
        #length = a.shape[0]
        #feature_vector = np.concatenate([a,b,c[length:],c[:length]],axis=1)
        #print(b.shape[0]/len(frames));exit()
        length = int(b.shape[0]/len(frames))
        feature_vectors = []
        for i in range(len(frames)):
            offset = 1*i*length   # scanning through the molecules
            #feature_vectors.append(np.concatenate([a[0+offset:length+offset],b[0+offset:length+offset],c[0+2*offset:length+2*offset],a[length+2*offset:2*length+2*offset]],axis=1))
            #feature_vectors.append(np.concatenate([a[0+offset:length+offset],b[0+offset:length+offset],c[0+offset:length+offset],a[length+offset:2*length+offset]],axis=1))
            average = np.mean(np.stack((a[offset:length+offset],a[length+offset:2*length+offset]),axis=2),axis=2)
            #print(offset)
            #print(a.shape)
            #print(average.shape)
            #print(b[0+offset:length+offset].shape)
            #print(c[0+offset:length+offset].shape)
            feature_vectors.append(np.concatenate([average,b[0+offset:length+offset],c[0+offset:length+offset]],axis=1))
        feature_vector = np.concatenate(feature_vectors)
    elif scrungus == 0:   # remove species consideration
        a = features.block(0).values.squeeze()
        #print(a.shape);exit()
        length = int(a.shape[0]/4/len(frames))
        feature_vectors = []
        for i in range(len(frames)):
            offset = 4*i*length
            #feature_vectors.append(np.concatenate([a[0+offset:length+offset],a[length+offset:2*length+offset],a[2*length+offset:3*length+offset],a[3*length+offset:4*length+offset]],axis=1))
            #average = np.mean(np.stack((a[length+offset:2*length+offset],a[2*length+offset:3*length+offset]),axis=2),axis=2)
            #feature_vectors.append(np.concatenate([average,a[3*length+offset:4*length+offset]],axis=1))
            average = np.mean(np.stack((a[2*length+offset:3*length+offset],a[3*length+offset:4*length+offset]),axis=2),axis=2)
            feature_vectors.append(np.concatenate([average,a[1*length+offset:2*length+offset]],axis=1))
        feature_vector = np.concatenate(feature_vectors)
    else:
        print(f'invalid scrungus value {scrungus}!!!')
        exit()
    return feature_vector



def anisaponify_frames(frames_dic,hypers,scrungus):
    calculator = EllipsoidalDensityProjection(**hypers)
    for struc in frames_dic:
        for t in frames_dic[struc]:
            frames = frames_dic[struc][t]['frames']
            frames_dic[struc][t]['features'] = anisaponify_frame(frames,calculator,scrungus)
            frames_dic[struc][t]['labels'] = frames_dic[struc][t]['labels']
    return frames_dic



def read_projects(projects,n_samples,t_begin,t_end,scrungus,frame_step=1000):
    i = 0
    frames_dic = {}
    for project in projects:
        if i==0:
            layered = 'layered'
            for job in project:
                job.doc.layered = 1
        else:
            layered = 'bulk'
            for job in project:
                job.doc.layered = 0
        frames_dic[layered] = {}
        frames_dic = read_frames(project,frames_dic,layered,n_samples,t_begin,t_end,scrungus,frame_step=1000)
        i += 1
    return frames_dic



def read_frames(project,frames_dic,struc,n_samples,t_begin,t_end,scrungus,frame_step=1000):
    #jobs = project.find_jobs()

    if n_samples < 0:
        n_samples = (t_end-t_begin)/job.sp.frame_step
    t_frames = np.linspace(t_begin,t_end,n_samples)
    t_frames = np.round(t_frames,-int(np.log10(frame_step))).astype(int)

    #frames_dic = {}
    #frames_dic[struc] = {}
    for t_frame in t_frames:
        frames_dic[struc][t_frame] = {'frames':[],'labels':[]}
        for job in project:
            fpath = job.fn(f'trajectory_prod/full_{job.sp.T}_{int(t_frame)}.xyz')
            #fpath = job.fn(f'trajectory_equil/full_{job.sp.T}_{int(t_frame)}.xyz')
            frame = ase.io.read(fpath)
            frame = frame_to_ellipsoids(frame,scrungus)
            frames_dic[struc][t_frame]['frames'].append(frame)
            frames_dic[struc][t_frame]['labels'].extend([job.doc.layered]*int(len(frame)/4))

    return frames_dic



def frame_to_ellipsoids(atoms,scrungus):
    center = np.array([0,1,16,17,28,29,47,55,65])
    paddle_A = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,56,57,58,59,60,61,62,63,64])
    paddle_a1 = np.array([18,19,20,21,22,23,24,25,26,27,40,41,42,43,44,45,46])
    paddle_a2 = np.array([30,31,32,33,34,35,36,37,38,39,48,49,50,51,52,53,54])

    #atoms = ase.io.read(fname)

    N = len(atoms)
    molecules = int(N/66)
    centers = []
    paddles_A = []
    paddles_a1 = []
    paddles_a2 = []
    ids = []
    if scrungus == 1:
        ids = [8,17,7,7]
    elif scrungus == 0:
        ids = [3,3,3,3]
    else:
        print(f'invalid scrungus value {scrungus}!!!')
        exit()
    for i in range(molecules):
        centers.extend(center+i*66)
        paddles_A.extend(paddle_A+i*66)
        paddles_a1.extend(paddle_a2+i*66)
        paddles_a2.extend(paddle_a2+i*66)

    TAB_dict = {
        "center": {
            "c_q": [],
            "pos": [],
            "shape": (2,2,0.5),
            "id": ids[0]
        },
        "paddle_A": {
            "c_q": [],
            "pos": [],
            "shape": (6,2,0.5),
            "id": ids[1]
        },
        "paddle_a1": {
            "c_q": [],
            "pos": [],
            "shape": (4,2,0.5),
            "id": ids[2]
        },
        "paddle_a2": {
            "c_q": [],
            "pos": [],
            "shape": (4,2,0.5),
            "id": ids[3]
        },
    }

    centers_ell = []
    centers_quat = []
    centers_pos = []
    x_orientation = []
    y_orientation = []
    z_orientation = []

    for i in range(molecules):
        for region_str, region_indices in zip(("center","paddle_A","paddle_a1","paddle_a2"), (center,paddle_A,paddle_a1,paddle_a2)):
            region = atoms[region_indices + i*66]
            region = region[region.numbers==6]

            vectors = region.get_all_distances(mic=True,vector=True)    # nxnx3
            com = region[0].position + vectors[0].mean(axis=0)
            mom,evecs = get_moments_of_inertia(region,vectors=True)
            if np.allclose(np.linalg.det(evecs),-1):
                evecs *= -1
            quat = R.from_matrix(evecs.T).as_quat()
            quat = np.roll(quat,1)

            TAB_dict[region_str]["pos"].append(com)
            TAB_dict[region_str]["c_q"].append(quat)

    ell_centers = Atoms(positions=TAB_dict["center"]["pos"],numbers=[TAB_dict["center"]["id"]]*molecules,cell=atoms.cell,pbc=atoms.pbc)
    ell_A = Atoms(positions=TAB_dict["paddle_A"]["pos"],numbers=[TAB_dict["paddle_A"]["id"]]*molecules,cell=atoms.cell,pbc=atoms.pbc)
    ell_a1 = Atoms(positions=TAB_dict["paddle_a1"]["pos"],numbers=[TAB_dict["paddle_a1"]["id"]]*molecules,cell=atoms.cell,pbc=atoms.pbc)
    ell_a2 = Atoms(positions=TAB_dict["paddle_a2"]["pos"],numbers=[TAB_dict["paddle_a2"]["id"]]*molecules,cell=atoms.cell,pbc=atoms.pbc)

    for region_str,ell in zip(("center","paddle_A","paddle_a1","paddle_a2"),(ell_centers,ell_A,ell_a1,ell_a2)):
        ell.arrays["c_q"] = np.asarray(TAB_dict[region_str]["c_q"])
        ell.arrays["c_diameter[1]"] = TAB_dict[region_str]["shape"][0]*np.ones(molecules)
        ell.arrays["c_diameter[2]"] = TAB_dict[region_str]["shape"][1]*np.ones(molecules)
        ell.arrays["c_diameter[3]"] = TAB_dict[region_str]["shape"][2]*np.ones(molecules)

    final_ells = ell_centers+ell_A+ell_a1+ell_a2
    return final_ells



def subsample_frames_dic(frames_dic,n_subsample):
    features = []
    labels = []
    ts = []
    for struc in frames_dic:
        for t in frames_dic[struc]:
            tmp_features = frames_dic[struc][t]['features'].copy()
            frames_dic[struc][t]['features'] = []
            frames_dic[struc][t]['labels'] = []
            frames_dic[struc][t]['indices'] = []
            for features in tmp_features:
                idx = np.cumsum(np.ones(features.shape[0])).astype(int) - 1
                idx = np.random.choice(idx,size=n_subsample,replace=False)
                frames_dic[struc][t]['features'].append(features[idx])
                frames_dic[struc][t]['labels'].append(struc*np.ones(features[idx].shape[0]).astype(int))
                frames_dic[struc][t]['indices'].append(idx)


    return frames_dic



def fit_scaling(frames_dic):
    #df = pd.DataFrame(np.vstack(list(features_dic.values())))    # we flatten along struc to get dimension: len(strucs)*n_samples*n_particles x feature_length
    df = pd.DataFrame(np.vstack([vector for struc in frames_dic for t in frames_dic[struc] for vector in frames_dic[struc][t]['features']]))
    scaling = StandardFlexibleScaler(column_wise=False)
    scaling.fit(df)
    return scaling.mean_,scaling.scale_



def transform_scaling(frames_dic,u,s):
    for struc in frames_dic:
        for t in frames_dic[struc]:
            frames_dic[struc][t]['features'] = (frames_dic[struc][t]['features']-u)/s
    return frames_dic



def fit_pcovc(frames_dic,n_components,alpha=0.5):
    #pcovc = PCovC(mixing=alpha,n_components=n_components,classifier=LogisticRegression())
    pcovc = PCovC(mixing=alpha,n_components=n_components)
    scaled_features = np.array([feature for struc in frames_dic for t in frames_dic[struc] for feature in frames_dic[struc][t]['features']])
    #labels = np.array([struc for struc in scaled_features_dic for t in scaled_features_dic[struc] for feature in scaled_features_dic[struc][t]])
    labels = np.array([struc for struc in frames_dic for t in frames_dic[struc] for feature in frames_dic[struc][t]['features']])#[:,np.newaxis]
    #labels = np.reshape(labels,(-1,1))
    print(f'alpha={alpha}')
    pcovc.fit(scaled_features,labels)
    print(f'explained variance: '
          +''.join([f'\nPCov {i+1}: {pcovc.explained_variance_[i]}' for i in range(len(pcovc.explained_variance_))])
          +f'\nTotal:  {np.sum(pcovc.explained_variance_)}')
    
    return pcovc



def transform_pcovc(frames_dic,pcovc):
    for struc in frames_dic:
        for t in frames_dic[struc]:
            frames_dic[struc][t]['pcov'] = pcovc.transform(frames_dic[struc][t]['features'])
            #frames_dic[struc][t]['pcov'] = []
            #for feature in frames_dic[struc][t]['features']:
                #frames_dic[struc][t]['pcov'].append(pcovc.transform(feature))
    return frames_dic



def fit_gaussian_mle(data):
    data = np.array(data)
    mu = np.mean(data,axis=0)[:,np.newaxis]
    Sigma = np.cov(data.T)
    return mu,Sigma



def fit_lda(frames_dic):
    scaled_features = np.array([feature for struc in frames_dic for t in frames_dic[struc] for feature in frames_dic[struc][t]['features']])
    labels = np.array([struc for struc in frames_dic for t in frames_dic[struc] for feature in frames_dic[struc][t]['features']])#[:,np.newaxis]
    lda = LinearDiscriminantAnalysis()
    lda.fit(scaled_features,labels)
    return lda



def transform_lda(frames_dic,lda):
    for struc in frames_dic:
        for t in frames_dic[struc]:
            frames_dic[struc][t]['lda'] = lda.transform(frames_dic[struc][t]['features'])
    return frames_dic



def grid_gaussian(grid,mu,Sigma):
    det = np.linalg.det(Sigma)
    output = (2*np.pi*det)**-1 * np.exp(np.einsum('ij,jk,ki->i',-0.5*(grid-mu).T, np.linalg.inv(Sigma), (grid-mu)))
    grid_resolution = round(np.sqrt(grid.shape[1]))
    output = np.reshape(output,(grid_resolution,grid_resolution))
    return output



def init_grid(x_min,x_max,y_min,y_max,grid_resolution=500):
    grid = np.meshgrid(np.linspace(x_min,x_max,grid_resolution),np.linspace(y_min,y_max,grid_resolution))
    return grid



def plot_gaussian(fig,ax,grid,mu,Sigma,color,label):
    xedges = grid[0][0,:]
    yedges = grid[1][:,0]
    data = grid_gaussian(np.reshape(np.array(grid),(2,-1)),mu,Sigma)
    if color == '':
        #pcm = ax.pcolormesh(xedges,yedges,data)
        color = 'k'
        ax.contourf(xedges,yedges,data,colors=color)
    else:
        ax.contour(xedges,yedges,data,colors=color)
    patch = mpatches.Patch(color=color,label=label)
    return patch



def plot_scatter(fig,ax,data,color,label,n_samples=100):
    data = np.array(data)
    idx = np.cumsum(np.ones(data.shape[0])).astype(int) - 1
    idx = np.random.choice(idx,size=n_samples,replace=False)
    ax.scatter(data[idx,0],data[idx,1],c=color,s=10.0,alpha=0.5,label=label)



def plot_hists(fig,ax_histx,ax_histy,xmin,xmax,ymin,ymax,data,color,bins=50):
    hist_params = dict(histtype='stepfilled', color=color, edgecolor='k', bins=bins, density=True, alpha=0.6)
    #scaler = MinMaxScaler().fit(np.vstack((xedges,yedges)))
    ax_histx.hist(data[:,0],**hist_params)
    ax_histx.set_xlim(xmin,xmax)
    ax_histx.axis('off')
    ax_histy.hist(data[:,1],orientation='horizontal',**hist_params)
    ax_histy.set_ylim(ymin,ymax)
    ax_histy.axis('off')



def plot_kde(fig,ax,data,label,hist_params,color):
    data = pd.DataFrame(data)
    #kdeplot = sns.kdeplot(data=data,ax=ax,x=0,y=1,**hist_params)
    kdeplot = sns.kdeplot(data=data,ax=ax,x=0,y=1,color=color)
    patch = mpatches.Patch(color=color,label=label)
    return patch
    

def plot_gaussian_figs(frames_dic,alpha,x_min,x_max,y_min,y_max,grid_resolution=500):
    grid = init_grid(x_min,x_max,y_min,y_max,grid_resolution=grid_resolution)
    gaussian_dic = {}
    label = ''
    patches = []
    #fig, ax = plt.subplots(figsize=(6,6))
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['center', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='tight',
                              gridspec_kw={
                                  "hspace": 0,
                                  "wspace": 0,
                                  }
                              )
    for struc in frames_dic:
        if struc=='layered':
            label = 'Layered'
            color = 'r'
        else:
            label = 'Isotropic'
            color = 'b'
        gaussian_dic[struc] = {}
        pcovs = []
        for t in frames_dic[struc]:
            pcovs.append(frames_dic[struc][t]['pcov'][:,:2])
        pcovs = np.vstack(pcovs)

        #mu,Sigma = fit_gaussian_mle(frames_dic[struc][t]['pcov'][:,:2])
        mu,Sigma = fit_gaussian_mle(pcovs)
        #print(struc);print(mu);print(Sigma);print(np.linalg.det(Sigma))
        patch = plot_gaussian(fig,axs['center'],grid,mu,Sigma,color,label)
        #plot_scatter(fig,axs['center'],pcovs,color,label,n_samples=35)
        #plot_hists(fig,axs['histx'],axs['histy'],x_min,x_max,y_min,y_max,frames_dic[struc][t]['pcov'][:,:2],color)
        plot_hists(fig,axs['histx'],axs['histy'],x_min,x_max,y_min,y_max,pcovs,color)

        gaussian_dic[struc][0] = {}
        gaussian_dic[struc][0]['mu'] = mu
        gaussian_dic[struc][0]['Sigma'] = Sigma

        patches.append(patch)
            
    axs['center'].set_xlabel('PCov 1')
    axs['center'].set_ylabel('PCov 2')
    #axs['center'].set_xlim(-1.1,1.1)
    axs['center'].set_ylim(y_min,y_max)
    axs['center'].legend(handles=patches)
    #axs.set_title(f'Gaussian Fit of Structure Distribution, struc={struc}, t={t}')
    plt.savefig(f'gaussian_plots/gaussian_diff_{alpha}.png')
    plt.close()

    #print(ttest_ind(test['layered'],test['bulk'],equal_var=False))
    #exit()

    return gaussian_dic



def plot_total_gaussian_figs(frames_dic,alpha,x_min,x_max,y_min,y_max,grid_resolution=500):
    grid = init_grid(x_min,x_max,y_min,y_max,grid_resolution=grid_resolution)
    gaussian_dic = {}
    fig, ax = plt.subplots(figsize=(6,6))
    for struc in frames_dic:
        gaussian_dic[struc] = {}
        pcovs = []
        for t in frames_dic[struc]:
            pcovs.append(frames_dic[struc][t]['pcov'][:,:2])
    pcovs = np.vstack(pcovs)

    mu,Sigma = fit_gaussian_mle(frames_dic[struc][t]['pcov'][:,:2])
    plot_gaussian(fig,ax,grid,mu,Sigma,'','total')

    gaussian_dic[struc][0] = {}
    gaussian_dic[struc][0]['mu'] = mu
    gaussian_dic[struc][0]['Sigma'] = Sigma
            
    ax.set_xlabel('PCov 1')
    ax.set_ylabel('PCov 2')
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    #axs.set_title(f'Gaussian Fit of Structure Distribution, struc={struc}, t={t}')
    plt.savefig(f'gaussian_plots/gaussian_total_{alpha}.png')
    plt.close()

    return gaussian_dic



def plot_gaussian_mean(gaussian_dic,alpha):
    fig, ax = plt.subplots(figsize=(6,6))
    strucs = sorted(list(gaussian_dic.keys()))
    for struc in strucs:
        x = []
        y = []
        for t in gaussian_dic[struc]:
            x.append(gaussian_dic[struc][t]['mu'][0][0])
            y.append(gaussian_dic[struc][t]['mu'][1][0])
        ax.plot(x,y,label=f'{struc}')
    
    ax.legend()
    ax.set_xlabel('PCov 1')
    ax.set_ylabel('PCov 2')
    ax.set_title('Path of Distribution means')
    plt.savefig(f'gaussian_plots/mean_{alpha}.png')



def plot_gaussian_covariance(gaussian_dic,alpha):
    fig, ax = plt.subplots(figsize=(6,6))
    strucs = sorted(list(gaussian_dic.keys()))
    for struc in strucs:
        x = []
        y = []
        for t in gaussian_dic[struc]:
            x.append(t)
            y.append(np.linalg.det(gaussian_dic[struc][t]['Sigma']))
        ax.plot(x,y,label=f'{struc}')

    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('Determinant')
    ax.set_title('Determinant of Covariance')
    plt.savefig(f'gaussian_plots/covariance_{alpha}.png')



def plot_kde_fig(frames_dic,alpha,x_min,x_max,y_min,y_max):
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['center', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='tight',
                              gridspec_kw={
                                  "hspace": 0,
                                  "wspace": 0,
                                  }
                              )
    for struc in frames_dic:
        if struc=='layered':
            label = 'Layered'
            color = 'r'
        else:
            label = 'Isotropic'
            color = 'b'
        pcovs = []
        for t in frames_dic[struc]:
            pcovs.append(frames_dic[struc][t]['pcov'][:,:2])
        pcovs = np.vstack(pcovs)
        hist_params = dict(color=color, label=struc)
        plot_kde(fig,axs['center'],pcovs,struc,hist_params,color)
        #plot_scatter(fig,axs['center'],frames_dic[struc][t]['pcov'][:,:2],color,label,n_samples=35)
        plot_hists(fig,axs['histx'],axs['histy'],x_min,x_max,y_min,y_max,pcovs,color)
            
    axs['center'].set_xlabel('PCov 1')
    axs['center'].set_ylabel('PCov 2')
    axs['center'].set_xlim(x_min,x_max)
    axs['center'].set_ylim(y_min,y_max)
    axs['center'].legend()
    #axs.set_title(f'Gaussian Fit of Structure Distribution, struc={struc}, t={t}')
    plt.savefig(f'kde_plots/kde_diff_{alpha}.png')
    plt.close()



def plot_scatter_figs(frames_dic,alpha,n_samples,x_min,x_max,y_min,y_max):
    fig,ax = plt.subplots()
    for struc in frames_dic:
        if struc=='layered':
            label = 'Layered'
            color = 'r'
        else:
            label = 'Isotropic'
            color = 'b'
        pcovs = []
        for t in frames_dic[struc]:
            pcovs.append(frames_dic[struc][t]['pcov'][:,:2])
        pcovs = np.vstack(pcovs)
        #plot_kde(fig,ax,pcovs)
        plot_scatter(fig,ax,pcovs,color,label,n_samples=n_samples)
        #plot_hists(fig,axs['histx'],axs['histy'],x_min,x_max,y_min,y_max,pcovs,color)
            
    ax.set_xlabel('PCov 1')
    ax.set_ylabel('PCov 2')
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.legend()
    #axs.set_title(f'Gaussian Fit of Structure Distribution, struc={struc}, t={t}')
    plt.savefig(f'scatter_plots/scatter_diff_{alpha}.png')
    plt.close()



def plot_lda_fig(frames_dic,bins):
    fig,ax = plt.subplots()
    for struc in frames_dic:
        if struc=='layered':
            label = 'Layered'
            color = 'r'
        else:
            label = 'Isotropic'
            color = 'b'
        ldas = []
        for t in frames_dic[struc]:
            ldas.append(frames_dic[struc][t]['lda'])
        #ldas = np.vstack(ldas)
        ldas = np.vstack(ldas)
        print(ldas.shape)
        hist_params = dict(histtype='stepfilled', color=color, edgecolor='k', density=True, alpha=0.6, label=struc)
        #scaler = MinMaxScaler().fit(np.vstack((xedges,yedges)))
        ax.hist(ldas,bins=bins,**hist_params)

    ax.set_xlabel('Discriminant')
    ax.set_ylabel('Probability')
    #ax.set_xlim(x_min,x_max)
    #ax.set_ylim(y_min,y_max)
    ax.legend()
    #axs.set_title(f'Gaussian Fit of Structure Distribution, struc={struc}, t={t}')
    plt.savefig(f'lda_plots/lda_diff.png')
    plt.close()

    
    

def set_anisoap_hypers(lmax=5,nmax=3,cutoff_radius=7.0,gaussian_width=1.5):
    AniSOAP_HYPERS = {
        "max_angular": lmax,
        "max_radial": nmax,
        "radial_basis_name": "gto",
        "rotation_type": "quaternion",
        "rotation_key": "c_q",
        "cutoff_radius": cutoff_radius,
        "radial_gaussian_width": gaussian_width,
        "basis_rcond": 1e-8,
        "basis_tol": 1e-4,
        "compute_gradients": False,
        "subtract_center_contribution": False,
    }
    return AniSOAP_HYPERS



def plot_orientations(points,orientations):
    L = np.mean(points,axis=0)
    print(L)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver3D(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        orientations[:, 0],
        orientations[:, 1],
        orientations[:, 2],
        normalize=True,
        color="k",
    )

    nematic = Nematic()
    nematic.compute(orientations)
    ax.set_title(
        f"Nematic order parameter: {nematic.order:.2f} \n Director: {nematic.director}"
    )
    # plot the director as a blue arrow
    """
    ax.quiver3D(
        L[0] * nematic.order,
        L[1] * 1.2,
        L[2],
        nematic.director[0],
        nematic.director[1],
        nematic.director[2],
        length=10 * nematic.order,
        normalize=True,
        color="blue",
    )
    """
    plt.show()




def make_permol_chemiscope(frames_dic,cutoff_radius,scrungus,n_samples):
    traj = []
    structures = []
    timesteps = []
    pcovs = []
    q_strucs = []
    qs = []
    nematic = Nematic()
    test = {"layered":[],"bulk":[]}
    for struc in frames_dic:
        print(struc)
        for t in frames_dic[struc]:
            j = 0
            for frame,q_struc in zip(frames_dic[struc][t]['frames'],frames_dic[struc][t]['nematic']):
                N_mol = int(len(frame)/4)
                sample_indices = np.random.choice(np.cumsum(np.ones(N_mol))-1,size=n_samples,replace=False).astype(int)
                distances = frame.get_all_distances(mic=True, vector=False)
                
                for i in sample_indices:
                    q_strucs.append(q_struc)
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
                    frame_dic = mol_frame.todict()
                    diameters = frame_dic['c_diameter[1]']
                    quats = frame_dic['c_q']
                    rotations = R.from_quat(quats,scalar_first=True)
                    vecs = np.zeros((quats.shape[0],3))
                    #vecs[:,2] = 1
                    vecs[diameters==2,2] = 1
                    vecs[diameters!=2,0] = 1
                    vecs = rotations.apply(vecs)
                    #ase.io.write('test.xyz',mol_frame)
                    #plot_orientations(mol_frame.positions,vecs)
                    q = nematic.compute(vecs).order
                    #first = np.where(mol_in_cutoff>=1)[0][0]
                    #mol_in_cutoff[first],mol_in_cutoff[i] = mol_in_cutoff[i],mol_in_cutoff[first]
                    #mol_frame = frame[mol_in_cutoff>=1]
                    traj.append(mol_frame)
                    structures.append(struc)
                    timesteps.append(t)
                    pcovs.append(frames_dic[struc][t]['pcov'][i+j])
                    qs.append(q)
                    test[struc].append(q)
                    
    n_frames = len(traj)

    print(np.mean(test['layered']),np.mean(test['bulk']))
    print(np.std(test['layered']),np.std(test['bulk']))
    """
    plt.hist(test['layered'],bins=20,density=True,alpha=0.7)
    plt.hist(test['bulk'],bins=20,density=True,alpha=0.7)
    plt.show()
    exit()
    """

    properties = {      # the data
        "PCov": {
            "target": "structure",
            "values": pcovs,
            "description": "PCA of per-atom representation of the structures",
        },
        "Structure": {
            "target": "structure",
            "values": structures,
            "description": "global B particle number fraction",
        },
        "Timestep": {
            "target": "structure",
            "values": timesteps,
            "description": "timestep when the structure was sampled",
        },
        "Local Nematic Order": {
            "target": "structure",
            "values": qs,
            "description": "",
        },
        "Global Nematic Order": {
            "target": "structure",
            "values": q_strucs,
            "description": "",
        },
    }
    
    environments = [   
        (i, j, cutoff_radius) for i in range(n_frames) for j in range(len(traj[i]))
    ]
    #print(environments)
    #print(len(environments))
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
        path=f"chemiscope_{scrungus}.json",
        frames=traj,
        properties=properties,
        environments=environments,
        shapes=shapes
    )
                    
    


def save_frames_dic(frames_dic,scrungus): 
    file = open(f'frames_dic_{scrungus}','wb')
    pickle.dump(frames_dic,file)
    file.close()



def read_frames_dic(scrungus):
    file = open(f'frames_dic_{scrungus}','rb')
    frames_dic = pickle.load(file)
    file.close()
    return frames_dic



def calc_Sz(frames_dic):
    print('def calc_Sz')
    for struc in frames_dic:
        print(struc)
        S_z_average = []
        for t in frames_dic[struc]:
            for frame in frames_dic[struc][t]['frames']:
                #print(frame)
                frame_dic = frame.todict()
                c_d = frame_dic['c_diameter[1]']
                quats = frame_dic['c_q']
                matrixs = R.from_quat(quats,scalar_first=True).as_matrix()
                L = int(matrixs.shape[0]/4)
                print(3*np.mean(matrixs[:L,2,:]**2,axis=0)/2-1.0/2)
    exit()
    return frames_dic



def calc_nematic(frames_dic):
    nematic = Nematic()
    for struc in frames_dic:
        for t in frames_dic[struc]:
            frames_dic[struc][t]['nematic'] = []
            for frame in frames_dic[struc][t]['frames']:
                frame_dic = frame.todict()
                diameters = frame_dic['c_diameter[1]']
                quats = frame_dic['c_q']
                rotations = R.from_quat(quats,scalar_first=True)
                vecs = np.zeros((quats.shape[0],3))
                vecs[diameters==2,2] = 1
                vecs[diameters!=2,0] = 1
                vecs = rotations.apply(vecs)
                q = nematic.compute(vecs).order
                frames_dic[struc][t]['nematic'].append(q)
    return frames_dic



def correlation_length(frames_dic,bins=50):
    r_max = 9999
    j = 0
    for struc in frames_dic:
        distances = []
        correlations = []
        pcovs = []
        for t in frames_dic[struc]:
            for frame in frames_dic[struc][t]['frames']:
                frame_dic = frame.todict()
                diameters = frame_dic['c_diameter[1]']
                quats = frame_dic['c_q'][diameters!=2]
                vecs = np.zeros((quats.shape[0],3))
                vecs[:,2] = 1
                rotations = R.from_quat(quats,scalar_first=True)
                vecs = rotations.apply(vecs)
                corrs = 3.0/2*np.einsum('ij,kj',vecs,vecs)**2 - 1.0/2
                for i in range(len(frame[diameters!=2])):
                    dist = frame[diameters!=2].get_distances(i,None,mic=True)
                    distances.extend(dist)
                    correlations.extend(corrs[i,:])
                r_max = min(r_max,np.min(np.diag(frame.cell))/2)
                
        hist,edges = np.histogram(distances,weights=correlations,bins=bins)
        counts,garb = np.histogram(distances,bins=bins)
        hist = hist/counts
        plt.plot(edges[:-1],hist,label=f'{struc}')
        j += 1
    plt.xlim([0,r_max])
        
    plt.legend()
    plt.savefig('test_corr.png')
    #plt.show()
    #exit()



def setup_scaling_and_pcovc(projects,n_samples,cutoff_radius,l_max,n_max,n_subsample=int(1e5),n_components=2,alpha=0.05,t_begin=0,t_end=int(1e6),scrungus=0):
    #frames_dic = sample_frames(project,strucs,n_samples)    # frames_dic[struc][t]['frames'] = list of frames, we will flatten the t dimension of frames_dic for fitting
    frames_dic = read_projects(projects,n_samples,t_begin,t_end,scrungus,frame_step=1000)
    """
    i = 0
    frames_dic = {}
    for project in projects:
        if i==0:
            layered = 'layered'
            for job in project:
                job.doc.layered = 1
        else:
            layered = 'bulk'
            for job in project:
                job.doc.layered = 0
        frames_dic[layered] = {}
        frames_dic = read_frames(project,frames_dic,layered,n_samples,t_begin,t_end,scrungus,frame_step=1000)
        i += 1
    """
    #flat_frames_dic = {struc:{0:{'frames':[frame for t in frames_dic[struc] for frame in frames_dic[struc][t]['frames']],'identities':[frame for t in frames_dic[struc] for frame in frames_dic[struc][t]['identities']]}} for struc in frames_dic}
    hypers = set_anisoap_hypers(cutoff_radius=cutoff_radius,lmax=l_max,nmax=n_max)
    frames_dic = anisaponify_frames(frames_dic,hypers,scrungus)     # features_dic[struc] = numpy array of features
    correlation_length(frames_dic,bins=200)
    #frames_dic = subsample_frames_dic(frames_dic,n_subsample)
    u,s = fit_scaling(frames_dic)
    frames_dic = transform_scaling(frames_dic,u,s)
    pcovc = fit_pcovc(frames_dic,n_components,alpha=alpha)
    frames_dic = transform_pcovc(frames_dic,pcovc)
    lda = fit_lda(frames_dic)
    
    x_min = min([value for structure in frames_dic for t in frames_dic[structure] for value in frames_dic[structure][t]['pcov'][:,0]])
    y_min = min([value for structure in frames_dic for t in frames_dic[structure] for value in frames_dic[structure][t]['pcov'][:,1]])
    x_max = max([value for structure in frames_dic for t in frames_dic[structure] for value in frames_dic[structure][t]['pcov'][:,0]])
    y_max = max([value for structure in frames_dic for t in frames_dic[structure] for value in frames_dic[structure][t]['pcov'][:,1]])

    
    return pcovc,lda,u,s,x_min,x_max,y_min,y_max



def calc_frames_dic(projects,pcovc,lda,scrungus,cutoff_radius,u,s,n_samples,n_subsamples,t_begin=0,t_end=int(1e6),l_max=5,n_max=3,grid_resolution=500):
    strucs = ['layered','bulk']
    hypers = set_anisoap_hypers(lmax=l_max,nmax=n_max,cutoff_radius=cutoff_radius)
    frames_dic ={}
    frames_dic = read_projects(projects,n_samples,t_begin,t_end,scrungus,frame_step=1000)
    frames_dic = anisaponify_frames(frames_dic,hypers,scrungus)
    #frames_dic = subsample_frames_dic(frames_dic,n_subsamples)    # subsample 1000 particles per frame
    #frames_dic = calc_Sz(frames_dic)
    #frames_dic = count_neighbors(frames_dic,cutoff_radius)
    frames_dic = transform_scaling(frames_dic,u,s)
    frames_dic = transform_pcovc(frames_dic,pcovc)
    frames_dic = calc_nematic(frames_dic)
    frames_dic = transform_lda(frames_dic,lda)
    return frames_dic
