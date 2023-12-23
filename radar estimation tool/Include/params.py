from .generating import dp_numberOfEnabledChan, dp_getNumberOfFrameFromBinFile
import math
import numpy as np

C = 3e8

## --------- ADC DATA PARAM --------------
def dp_printADCDataParams(adcDataParams):
    """
    Description:    This function prints ADC raw data Parameters
    Input:          adcDataParams
    Output:         None
    """
    print('Input ADC data parameters:')
    print('\t dataFmt:{}'.format(adcDataParams['dataFmt']))
    print('\t iqSwap:{}'.format(adcDataParams['iqSwap']))
    print('\t chanInterleave:{}'.format(adcDataParams['chanInterleave']))
    print('\t numChirpsPerFrame:{}'.format(adcDataParams['numChirpsPerFrame']))
    print('\t adcBits:{}'.format(adcDataParams['adcBits']))
    print('\t numRxChan:{}'.format(adcDataParams['numRxChan']))
    print('\t numAdcSamples:{}'.format(adcDataParams['numAdcSamples']))

def dp_generateADCDataParams(mmwaveJSON, Params):
    """
    Description:    This function generates ADC raw data Parameters
    Input:          mmwaveJSON
    Output:         adcDataParams
    """
    adcDataParams = {}
    
    frameCfg = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']
    adcDataParams['dataFmt'] = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlAdcOutCfg_t']['fmt']['b2AdcOutFmt']
    adcDataParams['iqSwap'] = mmwaveJSON['mmWaveDevices'][0]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['iqSwapSel']
    adcDataParams['chanInterleave'] = mmwaveJSON['mmWaveDevices'][0]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['chInterleave']
    adcDataParams['numChirpsPerFrame']=frameCfg['numLoops']*(frameCfg['chirpEndIdx']-frameCfg['chirpStartIdx']+1)
    adcDataParams['adcBits'] = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlAdcOutCfg_t']['fmt']['b2AdcBits']
    rxChanMask = int(mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlChanCfg_t']['rxChannelEn'], 16)
   
    adcDataParams['numRxChan'] = dp_numberOfEnabledChan(rxChanMask)
    adcDataParams['numAdcSamples'] = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['numAdcSamples']

    
    dp_printADCDataParams(adcDataParams)

    # Calculate ADC data size
    try:
        if adcDataParams['adcBits'] == 2:
            if adcDataParams['dataFmt'] == 0:
                # real data, one sample is 16bits=2bytes
                gAdcOneSampleSize = 2
            elif (adcDataParams['dataFmt'] == 1) or (adcDataParams['dataFmt'] == 2):
                gAdcOneSampleSize = 4; # 2 bytes 
            else:
                raise ValueError("Error: unsupported ADC dataFmt")
        else:
          raise ValueError("Error: unsupported ADC bits ({})".format(adcDataParams['adcBits']))
    except ValueError as error:
        print(error)

    # data size per Chirp = sample size x the # of sample x the # of Rx channel
    dataSizeOneChirp = gAdcOneSampleSize * adcDataParams['numAdcSamples'] * adcDataParams['numRxChan']
    
    Params['dataSizeOneFrame'] = dataSizeOneChirp * adcDataParams['numChirpsPerFrame']
    Params['dataSizeOneChirp'] = dataSizeOneChirp
    Params['adcDataParams'] = adcDataParams

    return adcDataParams


## --------- RF PARAM --------------
def dp_generateRFParams(mmwaveJSON, radarCubeParams, adcDataParams):
    """
    Description:    This function generates mmWave Sensor RF parameters
    Input:          mmwaveJSON, radarCubeParams, adcDataParams
    Output:         RFParmas
    """
    RFParams = {}

    profileCfg = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']

    ## start Frequency = 77e8
    RFParams['startFreq'] = profileCfg['startFreqConst_GHz']

    # Slope const (MHz/usec)
    RFParams['freqSlope'] = profileCfg['freqSlopeConst_MHz_usec']

    # ADC sampling rate in Msps
    RFParams['sampleRate'] = profileCfg['digOutSampleRate'] / 1e3

    # Generate radarCube parameters
    RFParams['numRangeBins'] = 2 ** math.ceil(math.log2(adcDataParams['numAdcSamples']))
    RFParams['numDopplerBins'] = radarCubeParams['numDopplerChirps']
    RFParams['bandwidth'] = abs(RFParams['freqSlope'] * profileCfg['numAdcSamples'] / profileCfg['digOutSampleRate'])

    RFParams['rangeResolutionsInMeters'] = C * RFParams['sampleRate'] / (2 * RFParams['freqSlope'] * RFParams['numRangeBins'] * 1e6)
    RFParams['dopplerResolutionMps'] = C / (2 * RFParams['startFreq'] * 1e9 * (profileCfg['idleTimeConst_usec'] + profileCfg['rampEndTime_usec']) * 1e-6 * radarCubeParams['numDopplerChirps'] * radarCubeParams['numTxChan'])
    RFParams['framePeriodicity'] = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']['framePeriodicity_msec']
    
    # Antenna parameters

    return RFParams

## --------- RadarCube PARAM --------------
def dp_printRadarCubeParams(radarCubeParams):
    """
    Description:    This function prints radar cube data Matrix Parameters
    Input:          mmwaveJSON
    Output:         radarCubeParams
    """
    print('Radarcube parameters:')
    print('\t iqSwap:{}'.format(radarCubeParams["iqSwap"]))
    print('\t radarCubeFmt:{}'.format(radarCubeParams["radarCubeFmt"]))
    print('\t numDopplerChirps:{}'.format(radarCubeParams["numDopplerChirps"]))
    print('\t numRxChan:{}'.format(radarCubeParams["numRxChan"]))
    print('\t numTxChan:{}'.format(radarCubeParams["numTxChan"]))
    print('\t numRangeBins:{}'.format(radarCubeParams["numRangeBins"]))

def dp_generateRadarCubeParams(mmwaveJSON, Params):
    """
    Description:    This function generates radar cube data Matrix Parameters
    Input:          mmwaveJSON
    Output:         radarCubeParams
    """
    radarCubeParams = {}

    frameCfg = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']

    radarCubeParams['iqSwap'] = mmwaveJSON['mmWaveDevices'][0]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['iqSwapSel']
    rxChanMask = int(mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlChanCfg_t']['rxChannelEn'], 16)
    radarCubeParams['numTxChan'] = frameCfg['chirpEndIdx'] - frameCfg['chirpStartIdx'] + 1
    radarCubeParams['numRxChan'] = dp_numberOfEnabledChan(rxChanMask)

    radarCubeParams['numRangeBins'] = 2 ** math.ceil(math.log2(mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['numAdcSamples']))
    radarCubeParams['numDopplerChirps'] = mmwaveJSON['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']['numLoops']

     # 1D Range FFT output : cmplx16ImRe_t x[numChirps][numRX][numRangeBins] 
    radarCubeParams['radarCubeFmt'] = 1 #RADAR_CUBE_FORMAT_1;
    
    dp_printRadarCubeParams(radarCubeParams)
    Params['radarCubeParams'] = radarCubeParams

    return radarCubeParams


## --------- PARAMS --------------
def dp_generateParams(mmwaveJSON, binFileNames, Params, adcDataParams, radarCubeParams):
        
    Params['RFParams'] = dp_generateRFParams(mmwaveJSON, radarCubeParams, adcDataParams)
    Params['NSample'] = adcDataParams['numAdcSamples']
    Params['NChirp'] = adcDataParams['numChirpsPerFrame']
    Params['NChan'] = adcDataParams['numRxChan']
    Params['NTxAnt'] = radarCubeParams['numTxChan']
    Params['numRangeBins'] = radarCubeParams['numRangeBins']
    Params['numDopplerBins'] = radarCubeParams['numDopplerChirps']
    Params['rangeWinType']= 0
    
    # Validate the Capture configuration
    Params['numLane'] = dp_numberOfEnabledChan(int(mmwaveJSON['mmWaveDevices'][0]['rawDataCaptureConfig']['rlDevLaneEnable_t']['laneEn'], 16))
    Params['chInterleave'] = mmwaveJSON['mmWaveDevices'][0]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['chInterleave']
    
    # Open raw data from file
    Params['fid_rawData'] = []
    
    numBinFiles = Params['numBinFiles']
    try:
        for idx in range(numBinFiles):
            Params['fid_rawData'].append(open(binFileNames[idx], 'rb'))  # fid_rawData = 리스트로 가정, rb=이진 모드로 열겠다
            if Params['fid_rawData'][idx] == -1:
                print("Can not open Bin file {}, - {}".format(binFileNames[idx], "file open error"))
                raise ValueError('Quit with error')
    except ValueError as error:
            print(error)
            
    # Calculate number of Frames in bin File
    Params['NFrame'] = 0
    Params['NFramePerFile'] = []
    
    try:
        Params['NFramePerFile'].append(dp_getNumberOfFrameFromBinFile(binFileNames[idx], Params))
        Params['NFrame'] += Params['NFramePerFile'][idx]
        if len(Params['NFramePerFile']) == 0:
            raise ValueError("Not enough data in binary file") 
    except ValueError as error:
            print(error)
            

    # angle
    angle_size = 181
    angle_range = np.linspace(-np.pi/2,np.pi/2, angle_size).reshape(angle_size, 1)   # 실제 angle!! pt_angle하고는 다르다.

    Params['angle'] = {"angle size" : angle_size,
                       "angle range" : angle_range
                       }

if __name__ == '__main__':
    print(":)")