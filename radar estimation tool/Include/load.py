import numpy as np
import json

def load_json(Params, SETUP_JSON_FILE_NAME):
    """load json file

    Args:
        Params (dictionary): Params
        SETUP_JSON_FILE_NAME (string): load Json file name

    Raises:
        ValueError: bin file is not available

    Returns:
        [dictionary, dictionary, list]: Json content, mmWave content, bin file name list
    """
    
    binFileNames = []
    
    # Read configuration and setup files
    with open(SETUP_JSON_FILE_NAME, 'r') as setupFile:
        setupJSON = json.load(setupFile) # setupFile을 딕셔너리 형태로 setupJSON에 저장 --> setupJSON = 딕셔너리

    # Read mmwave JSON file
    jsonMmwaveFileName = setupJSON['configUsed']  # setupJSON 딕셔너리에서 configUsed 부분을 가져옴

    with open(jsonMmwaveFileName, 'r') as mmwaveFile:
        mmwaveJSON = json.load(mmwaveFile)  # mmwaveJSON = 딕셔너리 형태

    # Print parsed current system parameter
    print('mmwave Device: {}'.format(setupJSON['mmWaveDevice'])) # mmWaveDevice = "awr2243"

    # Read bin file name
    binFilePath = setupJSON['capturedFiles']['fileBasePath']
    numBinFiles = len(setupJSON['capturedFiles']['files'])
    ## validate the # of bin files
    try:
        if numBinFiles < 1:
            raise ValueError('Bin File is not available')
    except ValueError as error:
        print(error)

    Params['numBinFiles'] = numBinFiles 

    for idx in range(numBinFiles):
        file_name = binFilePath + '/' + setupJSON['capturedFiles']['files'][idx]['processedFileName']
        binFileNames.append(file_name) 
        
    return setupJSON, mmwaveJSON, binFileNames

# def loadFrameData(fid_rawData):
#      # Find the first byte of the frame
#     fid_rawData.seek(frameIdx * dataSizeOneFrame, 0)
#     print(fid_rawData)

#     try:
#         # Read in raw data in complex single
#         rawData = np.fromfile(fid_rawData, dtype=np.uint16, count= -1)[:dataSizeOneFrame//2]
#         if not len(rawData):
#             raise Exception("Error reading binary file")
#     except Exception as error:
#         print(error)

#     try:
#         if dataSizeOneFrame != len(rawData) * 2:
#             print(f"dp_loadOneFrameData, size = {len(rawData)}, expected = {dataSizeOneFrame // 2}")
#             raise Exception("Read data from bin file, have wrong length")
#     except Exception as error:
#         print(error)


#     rawDataComplex = dp_loadOneFrameData(Params['fid_rawData'][fidIdx], Params['dataSizeOneFrame'], frameIdx - currFrameIdx)

#     # Read in raw data in uint16
#     dataSet['rawDataUint16'] = rawDataComplex.astype(np.uint16)

#     return rawData

def dp_loadFrameData(fid_rawData, dataSizeOneFrame):
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
        total_num_frame, frame_size = len(raw) // (dataSizeOneFrame//2),  dataSizeOneFrame//2

        rawData = np.zeros((total_num_frame, frame_size), dtype=np.uint16)

        for _ in range(total_num_frame):
            rawData[_, :] = raw[_*frame_size: _*frame_size + frame_size]    # 우리에게 한 데이터는 2byte이니!!

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


def dp_loadOneFrameData(fid_rawData, dataSizeOneFrame, frameIdx):
    """
    Description:    This function load one frame data from binary file
    Input:          fid_rawData - fid for binary file
                    dataSizeOneFrame - size of one frame data
                    frameIdx - frame index
    Output:         rawData - one frame of raw ADC data
    """
    
    # Find the first byte of the frame
    fid_rawData.seek(frameIdx * dataSizeOneFrame, 0)

    try:
        rawData = np.fromfile(fid_rawData, dtype=np.uint16, count= -1)[:dataSizeOneFrame//2]

        if not len(rawData):
            raise Exception("Error reading binary file")
    except Exception as error:
        print(error)

    try:
        if dataSizeOneFrame != len(rawData) * 2:
            print(f"dp_loadOneFrameData, size = {len(rawData)}, expected = {dataSizeOneFrame // 2}")
            raise Exception("Read data from bin file, have wrong length")
    except Exception as error:
        print(error)

    return rawData

if __name__ == "__main__":
    print(':)')