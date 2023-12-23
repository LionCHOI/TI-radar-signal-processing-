from .generating import dp_updateFrameData

def dp_exportData(rawDataFileName, radarCubeDataFileName, Params, dataSet):
    """
    Description:    This function loads one frame data and perform range FFT
    Input:          exportRawDataFile - file name to export raw data
                    export1DFFTDataFile - file name to export 1D FFT data
    Output:         mat files
    """
    rawADCData, radarCubeData = [], []

    # Prepare data to be saved in numpy arrays
    if rawDataFileName or radarCubeDataFileName:
        for frameIdx in range(Params['NFrame']):
            dp_updateFrameData(frameIdx, Params, dataSet)
            rawADCData.append(dataSet['rawDataUint16'])
            radarCubeData.append(dataSet['radarCubeData'])

    # Export raw ADC data
    if rawDataFileName !='':
        adcRawData = {
            'rfParams': Params['RFParams'],
            'data': rawADCData,
            'dim': {
                'numFrames': Params['NFrame'],
                'numChirpsPerFrame': Params['adcDataParams']['numChirpsPerFrame'],
                'numRxChan': Params['NChan'],
                'numSamples': Params['NSample']
            }
        }
        # 이렇게 쓰면 adcRawData['dim'][0]['numFrames'] 같이 열을 지정해줘야함

        # 파일로 데이터 저장(.mat형식)
        #io.savemat(rawDataFileName, {'adcRawData': adcRawData}, format='7.3')


    # Export rangeFFT data
    if radarCubeDataFileName !='':
        radarCubeParams=Params['radarCubeParams']
        radarCube={
            'rfParams':Params['RFParams'],
            'data': radarCubeData,
            'dim': {
                'numFrames': Params['NFrame'],
                'numChirps': radarCubeParams['numTxChan']*radarCubeParams['numDopplerChirps'],
                'numRxChan': radarCubeParams['numRxChan'],
                'numRangeBins': radarCubeParams['numRangeBins'],
                'iqSwap':radarCubeParams['iqSwap']
            }
        }
        # Save params and data to mat file
        #io.savemat(radarCubeDataFileName, {'radarCube': radarCube}, format='7.3')

if __name__ == "__main__":
    print(":)")