import numpy as np
import matplotlib.pyplot as plt
import os
from .load import *
from .angle import getSteeringVector, music

MAX_RXCHAN = 4

def plot_init(Params, dataSet):
    # variable
    NChirp, NRangeBin, CFAR = Params['NChirp'], Params['numRangeBins'], Params['CFAR']
    pt_range, pt_velocity = dataSet['rangeBin'].flatten(), dataSet['dopplerBin'].flatten()

    # plt parameter
    plt.ion()
    fig = plt.figure(figsize=(8, 4), layout="constrained")    # The larger figsize, The slower speed & speed is slower when set the layout 
    gs = fig.add_gridspec(1, 2)
    ax_list, plt_list = [fig.add_subplot(gs[0, col]) for col in range(2)], []

    # range-doppler map
    if CFAR == True:
        plt_list.append(ax_list[0].pcolormesh(pt_velocity, pt_range, np.ones((NRangeBin, NChirp))))
    else:
        plt_list.append(ax_list[0].pcolormesh(pt_velocity, pt_range, np.ones((NRangeBin, NChirp)), vmax = 1e6))
    ax_list[0].set_xlabel('velocity')
    ax_list[0].set_ylabel('range')
    ax_list[0].set_title("range-doppler map")
    # fig.colorbar(plt_list[0], ax=ax_list[0])

    # range-angle map
    plt_list.append(ax_list[1].plot(0, 0, 'o')[0])
    ax_list[1].set_xlim([-50, 50])
    ax_list[1].set_ylim([0, 50])
    ax_list[1].set_xlabel('x-axis')
    ax_list[1].set_ylabel('y-axis')
    ax_list[1].set_title("range-angle map")

    plot = {
        'fig' : fig,
        'ax_list' : ax_list,
        'plt_list' : plt_list,
    }

    return plot

def radarCube_init(Params, dataSet, CFAR, PLOT):

    ## CFAR available
    Params['CFAR'] = CFAR
    Params['PLOT'] = PLOT

    ## generate windowing
    rangeWinType = Params['rangeWinType']

    if rangeWinType == 1:
        win = np.hanning((Params['NSample'])) # 윈도우 함수 종류(ex 바틀렛)
    elif rangeWinType == 2:
        win = np.blackman((Params['NSample']))
    else:
        win = np.ones((Params['NSample'])) 
    
    ## window size
    Params['windowing_size'] = win

    ## range scope for 1D range FFT
    numRangeBins, rangeResolutionsInMeters= Params['numRangeBins'], Params['RFParams']['rangeResolutionsInMeters']
    dataSet['rangeBin'] = np.linspace(0,  numRangeBins * rangeResolutionsInMeters, numRangeBins)
    
    ## velocity scope for 2D Doppler_range FFT
    numDopplerBins, dopplerResolutionsInMeters = Params['numDopplerBins'], Params['RFParams']['dopplerResolutionMps']
    dataSet['dopplerBin'] = np.linspace(-(numDopplerBins//2) * dopplerResolutionsInMeters,  (numDopplerBins//2) * dopplerResolutionsInMeters, numDopplerBins)

    ## steering vector for angle estimation
    angle_range = Params['angle']['angle range']
    dataSet['anlgeBin'] = angle_range

    x_axis_loc, y_axis_loc, z_axis_loc = np.sin(angle_range), np.cos(angle_range), np.zeros((len(angle_range), 1))
    pt_unit_dir_vecs_angle = np.concatenate((x_axis_loc, y_axis_loc, z_axis_loc), axis= 1)
    dataSet['s_vec_rx_angle'] = getSteeringVector(pt_unit_dir_vecs_angle)

def dp_reshape4LaneLVDS(rawData):
    """
    Description:    This function reshape raw data for 4 lane LVDS capture, 
                    Convert 4 lane LVDS data to one matrix
    Input:          rawData - raw ADC data from binary file
    Output:         frameData
    """
    rawData8 = rawData.reshape((len(rawData) // 8, 8)).transpose()
    rawDataI = rawData8[0:4, :].transpose().flatten()
    rawDataQ = rawData8[4:8, :].transpose().flatten()
    
    frameData = np.vstack((rawDataI, rawDataQ)).transpose()

    return frameData

def dp_numberOfEnabledChan(chanMask):
    """
    Description:    This function counts number of enabled channels from 
                    channel Mask.
    Input:          chanMask
    Output:         Number of channels
    """
    count = 0
    for chan in range(MAX_RXCHAN):
        bitVal = 2 ** chan
        if (chanMask & bitVal) == bitVal:
            count += 1
            chanMask -= bitVal
            if chanMask == 0:
                break

    return count

def dp_getNumberOfFrameFromBinFile(binFileName, Params):
    """
    Description:    This function calcultes number of frames of data available
                    in binary file
    Input:          binFileName - binary file name
    Output:         NFrame - number of Frames

    """
    try:
        fileSize = os.path.getsize(binFileName)
        NFrame = fileSize // Params['dataSizeOneFrame']
        if not fileSize:
            raise Exception('Reading Bin file failed')
    except Exception as error:
        print(error)
    
    return NFrame

def dp_generateFrameData(rawData, Params, dataSet):
    """
    Description:    This function reshape raw binary data based on capture
                   configuration, generates data in cell of 
                   [number of chirps, number of RX channels, number of ADC samples]
    Input:          rawData - raw ADC data
    Output:         frameData - reshaped ADC data
    """
    frameComplex = np.zeros((Params['NChan'], Params['NChirp'], Params['NSample']), dtype='complex')    
    
    if Params['numLane'] == 4:
        frameData = dp_reshape4LaneLVDS(rawData)

    # Checking iqSwap setting
    if Params['adcDataParams']['iqSwap'] == 1:
        # Data is in ReIm format, convert to ImRe format to be used in radarCube
        frameData[:, [0, 1]] = frameData[:, [1, 0]]

    # Convert data to complex: column 1 - Imag, 2 - Real
    frameCplx = frameData[:, 0] + 1j * frameData[:, 1]

    # Change Interleave data to non-interleave  --> 4개의 channel이 따로 구분되어 있으면 non, 하나로 모여있으면 interleave 
    if Params['chInterleave'] == 1:
        # Non-interleave data
        temp = frameCplx.reshape(Params['NSample'] * Params['NChan'], Params['NChirp']).T
        for chirp in range(Params['NChirp']):
            frameComplex[:, chirp, :] = temp[chirp, :].reshape((Params['NChan'], Params['NSample'])).T
    else:
        # Interleave data
        temp = frameCplx.reshape(Params['NChirp'], Params['NSample'] * Params['NChan'])
        for chirp in range(Params['NChirp']):
            frameComplex[:, chirp, :] = temp[chirp, :].reshape((Params['NSample'], Params['NChan'])).T
            
    # Save raw data    
    dataSet['rawFrameData'] = frameComplex
    
    return frameComplex



def processingChain_rangeFFT(Params, dataSet, frameIdx, COV_MAT, ANGLE_VALUES):
    """
    Description:    This function is part of processing Chain to perform 
                   range FFT operation.
    Input:          frameData in complex(time domain)
    Output:         radarCube
    """
    # variables    
        ## parameter
    n_antenna, NChirp, NChan, NRangeBin = Params['radarCubeParams']['numRxChan'], Params['NChirp'], Params['NChan'], Params['numRangeBins']
    win, CFAR, angle_size = Params['windowing_size'], Params['CFAR'], Params['angle']['angle size']

        ## data
    pt_range, pt_angle = dataSet['rangeBin'], dataSet['anlgeBin']
    s_vec_rx_angle = dataSet['s_vec_rx_angle']
    frameData = dataSet['rawFrameData']

    # range FFT
    range_FFT = np.fft.fft(frameData * win, NRangeBin, 2)
    dataSet['radarCubeData_1D_FFT_set'][frameIdx] = range_FFT

    # doppler FFT
    doppler_FFT = np.fft.fftshift(np.fft.fft(range_FFT, NChirp, 1), 1)

    if CFAR is True:
        Tr, Td, Gr, Gd, offset = 2, 1, 2, 1, 3.32

        dB_Doppler_fft = 10*np.log10(abs(doppler_FFT[0, :, :]))
        dB_Doppler_fft = dB_Doppler_fft/(dB_Doppler_fft.max())
        
        for range_idx in range(Tr+Gr, NRangeBin - (Gr+Tr)):
            for doppler_idx in range(Td+Gd, NChirp - (Gd+Td)):
                noise_level = 0
                for range_idx_2 in range(range_idx-(Tr+Gr),  range_idx+(Tr+Gr)+1):
                    for doppler_idx_2 in range(doppler_idx-(Td+Gd), doppler_idx+(Td+Gd)+1):
                        if (abs(range_idx-range_idx_2)> Gr or abs(doppler_idx-doppler_idx_2)>Gd):
                            noise_level += 10**(dB_Doppler_fft[doppler_idx_2, range_idx_2]/10)

                threshold = 10*np.log10(noise_level/(2*(Td+Gd+1)*2*(Tr+Gr+1)-(Gr*Gd)-1))
                threshold = threshold + offset

                CUT = dB_Doppler_fft[doppler_idx, range_idx]
                
                if (CUT < threshold):
                    dB_Doppler_fft[doppler_idx, range_idx] = 0
                else:
                    dB_Doppler_fft[doppler_idx, range_idx] = 1
            
        dB_Doppler_fft[(dB_Doppler_fft != 1) & (dB_Doppler_fft != 0)] = 0

    dataSet['radarCubeData_2D_FFT_set'][frameIdx] = doppler_FFT    

    # MUSIC
    ## finding existing index
    range_FFT_anlge = np.copy(range_FFT)
    
    if CFAR is True:
        doppler_idx, range_idx = np.where(np.abs(dB_Doppler_fft) == 1) 
    else:
        doppler_idx, range_idx = np.where(np.mean(np.abs(doppler_FFT), axis=0) >= 100000)
        ##### 특정 부분만 추출하기
        range_FFT_2 = np.copy(range_FFT)
        range_set = 99 # 19m

        range_interval = np.concatenate((np.where(range_idx == range_set-1)[0],
                                         np.where(range_idx == range_set)[0],
                                         np.where(range_idx == range_set+1)[0]))
        
        range_idx = range_idx[range_interval]

        COV_MAT[frameIdx] = range_FFT_2[:,0,range_set-1:range_set+1] @ range_FFT_2[:,0,range_set-1:range_set+1].conjugate().transpose()
        # print(np.real(COV_MAT[frameIdx])) # 확인 잘 나오는지
            
    find_num = len(range_idx)

    ## preprocessing the index
    THRESHLOD_COHERENCE_VELO = 1
    coherence_range_cnt = np.zeros((find_num), dtype='int64')

    for index in range(find_num):
        a = np.where(range_idx == range_idx[index])[0]
        if len(a) >= 2:
            for idx in (a-1):
                if np.abs(doppler_idx[idx] - doppler_idx[idx+1]) > THRESHLOD_COHERENCE_VELO:
                    coherence_range_cnt[index] += 1
    else:
        coherence_range_cnt[coherence_range_cnt == 0] = 1   # coherence가 없더라도 1을 더해줘야 한다.

    num_estimated_object = len(coherence_range_cnt)

    ## estimation
    THRESHLOD_COHERENCE_RANGE, NUM_SNAPSHOT, MAX_ESCAPE_CNT = 1, 20, 2       # range_Resolution을 결정한다. --> 이를 줄이면, resolution 증가하지만 동일거리에서 힘들다. 
    angle_estimate, CovMat = {}, np.zeros((n_antenna, n_antenna, NUM_SNAPSHOT)).astype('complex_') # complex가 중요하다.
    escape_cnt = 0

    for idx_estim in range(num_estimated_object):
        if (np.abs(range_idx[idx_estim] - range_idx[idx_estim - 1]) > THRESHLOD_COHERENCE_RANGE or 
            escape_cnt > MAX_ESCAPE_CNT or idx_estim == 0): # coherence range & first index is always pass
            
            for _ in range(NUM_SNAPSHOT):
                CovMat[:,:,_] = (range_FFT_anlge[:, _, range_idx[idx_estim]:range_idx[idx_estim]+1] @ 
                                    range_FFT_anlge[:, _, range_idx[idx_estim]:range_idx[idx_estim]+1].conj().transpose())

            final_CovMat = CovMat.mean(axis=2)
            DoAsMUSIC, psindB = music(final_CovMat, coherence_range_cnt[idx_estim], n_antenna, angle_size, s_vec_rx_angle)  # MUSIC algorithm

            angle_estimate[float(pt_range[range_idx[idx_estim]])] = pt_angle[DoAsMUSIC].flatten() # * 180 / np.pi
            # print(f'{idx_estim} check {pt_angle[DoAsMUSIC]}')   # 같은 거리에서 누구를 찾는 지를 확인할 수 있다.

            escape_cnt = 0
        else:
            escape_cnt += 1
            
    if 'DoAsMUSIC' in locals() and len(pt_angle[DoAsMUSIC]) != 0 :
        ANGLE_VALUES[frameIdx] = np.rad2deg(pt_angle[DoAsMUSIC])

    if frameIdx == COV_MAT.shape[0]:
        print(COV_MAT)
        print(ANGLE_VALUES)
        np.save("./data/output_COV", COV_MAT)
        np.save("./data/output_angle", ANGLE_VALUES)

    if Params['PLOT'] is True and len(angle_estimate) != 0:
        # 좌표 구하기
        func = lambda my_dict: np.array(([my_dict[0] * np.sin(my_dict[1])], [my_dict[0] * np.cos(my_dict[1])]))
        coord_estimate = np.concatenate(list(map(func, angle_estimate.items())), axis = 2).reshape(2, -1)

        fig, plt_list = Params['plot']['fig'], Params['plot']['plt_list']
        
        if CFAR is True:
            plt_list[0].set_array(dB_Doppler_fft.T)
        else:
            plt_doppler_FFT = 10 * np.log10(np.abs(np.transpose(doppler_FFT[0])))
            plt_list[0].set_clim(vmax = np.max(plt_doppler_FFT))
            plt_list[0].set_array(plt_doppler_FFT)

        plt_list[1].set_xdata(coord_estimate[0, :])
        plt_list[1].set_ydata(coord_estimate[1, :])

        fig.canvas.draw()
        fig.canvas.flush_events()

def dp_updateFrameData(frameIdx, Params, dataSet, COV_MAT, ANGLE_VALUES):
    """
    Description:    This function loads one frame data and perform range FFT
    Input:          frameIdx - frame index
    Output:         dataSet.rawFrameData(complex)
                    dataSet.radarCubeData(complex)
    """
    currFrameIdx, fidIdx = 0, 0 # Index for finding binFile    

    for idx in range(Params['numBinFiles']):
        if frameIdx < (Params['NFramePerFile'][idx] + currFrameIdx):
            fidIdx = idx
            break
        else:
            currFrameIdx += Params['NFramePerFile'][idx]

    if fidIdx < Params['numBinFiles']:
        # Load raw data from bin file
        # rawDataComplex = dp_loadOneFrameData(Params['fid_rawData'][fidIdx], Params['dataSizeOneFrame'], frameIdx - currFrameIdx)
        rawDataComplex = dataSet['rawDataComplex_set'][frameIdx]

        # Read in raw data in uint16
        dataSet['rawDataUint16'] = rawDataComplex.astype(np.uint16)

        # time domain data y value adjustments
        timeDomainData = rawDataComplex - (rawDataComplex >= 2**15) * 2**16
        
        # reshape data based on capture configurations
        dataSet['rawFrameData_set'][frameIdx] = dp_generateFrameData(timeDomainData, Params, dataSet)        
        
        # Perform rangeFFT        
        processingChain_rangeFFT(Params, dataSet, frameIdx, COV_MAT, ANGLE_VALUES)

        print(frameIdx)

if __name__ == "__main__":
    print(":)")

#######################################################################
def processall(Params, dataSet, CFAR, win):
    """
        Description:    This function is part of processing Chain to perform 
                    range FFT operation.
        Input:          frameData in complex(time domain)
        Output:         radarCube
        """
    # variables    
    n_antenna, NChirp, NChan, NRangeBin = Params['radarCubeParams']['numRxChan'], Params['NChirp'], Params['NChan'], Params['numRangeBins']
    win = Params['windowing_size']

    pt_range, pt_angle = dataSet['rangeBin'], dataSet['anlgeBin']

    angle_size, s_vec_rx_angle = Params['angle']['angle size'], dataSet['s_vec_rx_angle']

    frameData = dataSet['rawFrameData_set']

    # range FFT
    range_FFT = np.fft.fft(frameData * win, NRangeBin, 3)   # (frame, )
    dataSet['radarCubeData_1D_FFT_set'] = range_FFT

    # doppler FFT
    doppler_FFT = np.fft.fftshift(np.fft.fft(range_FFT, NChirp, 2), 1)
    if CFAR is True:
        Tr, Td, Gr, Gd, offset = 2, 1, 2, 1, 3.3

        dB_Doppler_fft = 10*np.log10(abs(doppler_FFT[:, 0, :, :]))
        dB_Doppler_fft = dB_Doppler_fft/(dB_Doppler_fft.max())
        
        for range_idx in range(Tr+Gr, NRangeBin - (Gr+Tr)):
            for doppler_idx in range(Td+Gd, NChirp - (Gd+Td)):
                noise_level = 0
                for range_idx_2 in range(range_idx-(Tr+Gr),  range_idx+(Tr+Gr)+1):
                    for doppler_idx_2 in range(doppler_idx-(Td+Gd), doppler_idx+(Td+Gd)+1):
                        if (abs(range_idx-range_idx_2)> Gr or abs(doppler_idx-doppler_idx_2)>Gd):
                            noise_level += 10**(dB_Doppler_fft[:, doppler_idx_2, range_idx_2]/10)

                threshold = 10*np.log10(noise_level/(2*(Td+Gd+1)*2*(Tr+Gr+1)-(Gr*Gd)-1))
                threshold = threshold + offset

                # CUT = dB_Doppler_fft[:, doppler_idx, range_idx]
                
                dB_Doppler_fft[dB_Doppler_fft[:, doppler_idx, range_idx] < threshold] = 0
                dB_Doppler_fft[dB_Doppler_fft[:, doppler_idx, range_idx] > threshold] = 1

            # print(range_idx)

                # if (CUT < threshold):
                    # dB_Doppler_fft[:, doppler_idx, range_idx] = 0
                # else:
                    # dB_Doppler_fft[:, doppler_idx, range_idx] = 1

        dB_Doppler_fft[(dB_Doppler_fft != 1) & (dB_Doppler_fft != 0)] = 0

    dataSet['radarCubeData_2D_FFT_set'] = doppler_FFT    

    # MUSIC
    plt.ion()
    fig = plt.figure(figsize=(10, 5), layout="constrained")   

    for frame_idx in range(Params['NFrame']):

        ## finding existing index
        range_FFT_anlge = np.copy(range_FFT[frame_idx])
        
        if CFAR is True:
            doppler_idx, range_idx = np.where(dB_Doppler_fft[frame_idx] == 1)
        else:
            doppler_idx, range_idx = np.where(doppler_FFT[frame_idx, 0, :, :] >= 15000)
    
        find_num = len(range_idx)
        # print(find_num) 

        ## preprocessing the index
        THRESHLOD = 5
        coherence_range_cnt = np.zeros((find_num), dtype='int64')

        for index in range(find_num):
            num_same_range = np.where(range_idx == range_idx[index])[0]  # 같은 거리에 개수
            if len(num_same_range) >= 2:
                for idx in (num_same_range-1):
                    if np.abs(doppler_idx[idx] - doppler_idx[idx+1]) > THRESHLOD:   # 다른 속도를 가진 물체의 수 세기 (속도가 다르면 다른 물체)
                        coherence_range_cnt[index] += 1
        else:
            coherence_range_cnt[coherence_range_cnt == 0] = 1   # coherence가 없더라도 1을 더해줘야 한다.

        num_estimated_object = len(coherence_range_cnt)

        ## estimation
        THRESHLOD_COHERENCE_RANGE, NUM_SNAPSHOT = 2, 20       # range_Resolution을 결정한다. --> 이를 줄이면, resolution 증가하지만 동일거리에서 힘들다. 
        angle_estimate, CovMat = {}, np.zeros((n_antenna, n_antenna, NUM_SNAPSHOT)).astype('complex_') # complex가 중요하다.

        for idx_estim in range(num_estimated_object):
            if (np.abs(range_idx[idx_estim] - range_idx[idx_estim - 1]) > THRESHLOD_COHERENCE_RANGE or idx_estim == 0): # coherence range & first index is always pass
                
                for _ in range(NUM_SNAPSHOT):
                    CovMat[:,:,_] = (range_FFT_anlge[:, _, range_idx[idx_estim]:range_idx[idx_estim]+1] @ 
                                        range_FFT_anlge[:, _, range_idx[idx_estim]:range_idx[idx_estim]+1].conj().transpose())

                final_CovMat = CovMat.mean(axis=2)
                DoAsMUSIC, psindB = music(final_CovMat, coherence_range_cnt[idx_estim], n_antenna, angle_size, s_vec_rx_angle)  # MUSIC algorithm

                angle_estimate[float(pt_range[range_idx[idx_estim]])] = pt_angle[DoAsMUSIC].flatten() # * 180 / np.pi
                # print(f'{idx_estim} check {pt_angle[DoAsMUSIC]}')   # 같은 거리에서 누구를 찾는 지를 확인할 수 있다.


        # 좌표 구하기
        func = lambda my_dict: np.array(([my_dict[0] * np.sin(my_dict[1])], [my_dict[0] * np.cos(my_dict[1])]))
        coord_estimate = np.concatenate(list(map(func, angle_estimate.items())), axis = 2).reshape(2, -1)

        plt.cla()

        plt.scatter(*coord_estimate)
        plt.xlim([-50, 50])
        plt.ylim([0, 50])
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title("estimate")

        fig.canvas.draw()
        fig.canvas.flush_events()