from .generating import dp_numberOfEnabledChan

SUPPORTEDPLATFORM = ['awr1642', 'iwr1642', 'awr1243', 'awr1443', 'iwr1443', 'awr1843', 'iwr1843', 'iwr6843', 'awr2243']

def dp_validateDataCaptureConf(setupJson, mmwaveJSON, Params):
    """
    Description:    This function validates configuration from JSON files
    Input:          setupJson - setup JSON configuration structure
                    mmwaveJSON - mmwave JSON configuration structure
    Output:         confValid - true if the configuration is valid

    """
    mmWaveDevice = setupJson['mmWaveDevice']
    confValid = True

    # Validate if the device is supported. 1 is valid, 0 is unvalid
    for i in SUPPORTEDPLATFORM:
        if i == mmWaveDevice:
            index = 1
        else:
            index = 0

    # Validate of platform (HW)
    if index == 0:
        print('Platform not supported : {}'.format(mmWaveDevice))
        confValid = False

    # Validate the captureHardware
    if setupJson['captureHardware'] != 'DCA1000':
        confValid = False
        print("Capture hardware is not supported : {}".format(setupJson['captureHardware']))
      
    # Validate ADC_ONLY capture  
    if mmwaveJSON['mmWaveDevices'][0]['rawDataCaptureConfig']['rlDevDataPathCfg_t']['transferFmtPkt0'] != '0x1':
        confValid = False
        print("Capture data format is not supported : {}", mmwaveJSON['mmWaveDevices']['rawDataCaptureConfig']['rlDevDataPathCfg_t']['transferFmtPkt0'])        
    
    # Validate the dataLoggingMode
    if setupJson['DCA1000Config']['dataLoggingMode'] != 'raw':
        confValid = False
        print("Capture data logging mode is not supported : {}", setupJson['DCA1000Config']['dataLoggingMode'])

    if mmWaveDevice in ['awr1443', 'iwr1443', 'awr1243', 'awr2243']:
        if Params['numLane'] != 4:
            print("{} LVDS Lane is not supported for device: {}".format(Params['numLane'],mmWaveDevice))
            confValid = False
        if Params['chInterleave'] != 0:
            print("Interleave mode {} is not supported for device : {}".format(Params['chInterleave'],mmWaveDevice))
            confValid = False
    else:
        if Params['numLane'] != 2:
            print("{} LVDS Lane is not supported for device: {}".format(Params['numLane'],mmWaveDevice))
            confValid = False

        if Params['chInterleave'] != 1:
            print("Interleave mode {} is not supported for device: {}".format(Params['chInterleave'],mmWaveDevice))
            confValid = False

    if confValid == False:
        print("Configuration from JSON file is not valid")
        
    return confValid

if __name__ == "__main__":
    print(":)")