import numpy as np
from copy import deepcopy
import numpy.linalg as LA               # for norm
import scipy.signal as ss

def getRotationMatrixFromVector(q):
    """ 
    compute 3x3 rotation matrix from quaternion 
    orientation of GCS relative to LCS (Android system)
    q = [cos(theta/2), ex sin(theta/2), ey sin(theta/2), ez sin(theta/2)]"""
    
    if q.size != 4:
        print('Dimension error')
        
    q = q / np.sqrt(sum(q**2))

    
    ## Rotation matrix from quaternion
    sq_q1 = 2 * q[1]**2
    sq_q2 = 2 * q[2]**2
    sq_q3 = 2 * q[3]**2
    q1_q2 = 2 * q[1] * q[2]
    q3_q0 = 2 * q[3] * q[0]
    q1_q3 = 2 * q[1] * q[3]
    q2_q0 = 2 * q[2] * q[0]
    q2_q3 = 2 * q[2] * q[3]
    q1_q0 = 2 * q[1] * q[0]

    rotMat = np.array([[1 - sq_q2 - sq_q3, q1_q2 - q3_q0, q1_q3 + q2_q0],
            [q1_q2 + q3_q0, 1 - sq_q1 - sq_q3, q2_q3 - q1_q0],
            [q1_q3 - q2_q0, q2_q3 + q1_q0, 1-sq_q1 - sq_q2]])
    
    return rotMat


def getSteeringVector(ray_vectors_ref_frame):
    
    h_n = 4
    v_n = 1
    
    h_spacing = 0.5
    v_spacing = 0.5    

    quat = np.array([0, 0, 0.7071, 0.7071])
    
    rotMat = getRotationMatrixFromVector(quat)

    ray_vectors_local_frame = rotMat.transpose() @ ray_vectors_ref_frame.transpose()
    ray_vectors_local_frame = ray_vectors_local_frame.transpose()

    # x_ref = ray_vectors_local_frame[:,0]
    # y_ref = ray_vectors_local_frame[:,1]
    # z_ref = ray_vectors_local_frame[:,2]

    ## Steering vector
    antenna_loc = np.arange(0, h_spacing*h_n, h_spacing).reshape(h_n, 1) + 1j*np.arange(0, v_spacing*v_n, v_spacing)

    if antenna_loc.size != 0:
        antenna_loc = antenna_loc[:] - antenna_loc[0]
    else:
        antenna_loc = [0, ]

    antenna_loc_real_imag = np.concatenate((np.real(antenna_loc).reshape(len(antenna_loc), 1), np.imag(antenna_loc)), axis = 1)
    antenna_loc = np.concatenate((antenna_loc_real_imag, np.zeros((len(antenna_loc), 1))), axis = 1)

    n_antenna_elements = antenna_loc.shape[0]  # check this part
    n_rays = ray_vectors_local_frame.shape[0]  # check this part

    for k in range(n_rays):
        phase_diff = 2 * np.pi * (antenna_loc @ np.transpose(ray_vectors_local_frame[k]).reshape(antenna_loc.shape[1],1))
        if 'steering_vec' not in locals():
            steering_vec = np.exp(-1j*phase_diff)
        else:
            steering_vec = np.concatenate((steering_vec, np.exp(-1j*phase_diff)), axis = 1)

    return steering_vec

def angle_init(Params, dataSet):
    ## steering vector
    angle_range = Params['angle']['angle range']

    x_axis_loc, y_axis_loc, z_axis_loc = np.sin(angle_range), np.cos(angle_range), np.zeros((len(angle_range), 1))
    pt_unit_dir_vecs_angle = np.concatenate((x_axis_loc, y_axis_loc, z_axis_loc), axis= 1)
    dataSet['s_vec_rx_angle'] = getSteeringVector(pt_unit_dir_vecs_angle)


def music(CovMat, n_pt, n_antenna, angle_size, s_vec_rx_angle):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    
    _, V = LA.eig(CovMat)
    Qn  = V[:, n_pt:n_antenna]

    pspectrum = np.zeros(angle_size)
    pspectrum = 1/LA.norm((Qn.conj().transpose() @ s_vec_rx_angle), axis=0)
    
    psindB       = np.log10(10*pspectrum/pspectrum.min())
    DoAsMUSIC, _ = ss.find_peaks(psindB,height=1.35, distance=1.5)
    return DoAsMUSIC, pspectrum
