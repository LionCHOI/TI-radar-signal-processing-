import numpy as np




def dp_loadOneFrameData(fid_rawData, dataSizeOneFrame, frameIdx):
    """
    Description:    This function load one frame data from binary file
    Input:          fid_rawData - fid for binary file
                    dataSizeOneFrame - size of one frame data
                    frameIdx - frame index
    Output:         rawData - one frame of raw ADC data
    """
    
    try:
        # Read in raw data in complex single
        raw = np.fromfile(fid_rawData, dtype=np.uint16, count= -1)
        total_num_frame, frame_size = len(raw) // dataSizeOneFrame,  dataSizeOneFrame//2

        rawData = np.zeros((total_num_frame, frame_size), dtype=np.uint16)

        for _ in range(total_num_frame):
            rawData[_, :] = raw[_*dataSizeOneFrame: _*dataSizeOneFrame + dataSizeOneFrame//2]

        if not len(rawData[0]):
            raise Exception("Error reading binary file")
    except Exception as error:
        print(error)

    try:
        if dataSizeOneFrame != len(rawData[0]) * 2:
            print(f"dp_loadOneFrameData, size = {len(rawData[0])}, expected = {dataSizeOneFrame // 2}")
            raise Exception("Read data from bin file, have wrong length")
    except Exception as error:
        print(error)

    return rawData






rawDataComplex = dp_loadOneFrameData(Params['fid_rawData'][fidIdx], Params['dataSizeOneFrame'], frameIdx - currFrameIdx)

timeDomainData = rawDataComplex - (rawDataComplex >= 2**15) * 2**16



def dp_reshape4LaneLVDS(rawData):

    rawData8 = rawData.reshape((len(rawData) // 8, 8)).transpose()
    rawDataI = rawData8[0:4, :].transpose().flatten()
    rawDataQ = rawData8[4:8, :].transpose().flatten()
    
    frameData = np.vstack((rawDataI, rawDataQ)).transpose()

    return frameData


def dp_generateFrameData(rawData, Params, dataSet):

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