from Include.load import *
from Include.generating import *
from Include.params import *
# from Include.validation import *
# from Include.export import *
from Include.ui import *
from Include.angle import *

## variables
SETUP_JSON_FILE_NAME = '22xx.setup.json'
RAW_DATA_FILE_NAME = 'mydata.mat'
RADAR_CUBE_DATA_FILE_NAME ='radarcube.mat'
DEBUG_PLOT = False # True = plot, False = no plot
CFAR = False

Params, dataSet = {}, {}

# load JSON for setting parameter
setupJSON, mmwaveJSON, binFileNames = load_json(Params, SETUP_JSON_FILE_NAME)

# Generate ADC data parameters
adcDataParams = dp_generateADCDataParams(mmwaveJSON, Params)

# Generate radar cube parameters
radarCubeParams = dp_generateRadarCubeParams(mmwaveJSON, Params)

# Generate RF parameters
dp_generateParams(mmwaveJSON, binFileNames, Params, adcDataParams, radarCubeParams)

# initalize
radarCube_init(Params, dataSet, CFAR) # set the paramter for estimation

dataSet['rawDataComplex_set'] = dp_loadFrameData(Params['fid_rawData'][0], Params['dataSizeOneFrame'])
dataSet['rawFrameData_set'] = np.zeros((Params['NFrame'], Params['NChan'], Params['NChirp'], Params['NSample']), dtype='complex')
dataSet['radarCubeData_1D_FFT_set'] = np.zeros((Params['NFrame'], Params['NChan'], Params['NChirp'], Params['NSample']), dtype='complex')
dataSet['radarCubeData_2D_FFT_set'] = np.zeros((Params['NFrame'], Params['NChan'], Params['NChirp'], Params['NSample']), dtype='complex')

Params['plot'] = plot_init(Params, dataSet) # plot

# processing
for frame_Idx in range(Params['NFrame']):
    dp_updateFrameData(frame_Idx, Params, dataSet)

# Start example UI and update time domain/range Profile plots
if DEBUG_PLOT:
    # Plot the result
    try:
        for chanIdx in range(Params['NChan']):
            if is_pressed('esc'):
                chanIdx = int(input("Which channel do you wanna see? "))
                if chanIdx < 0 or chanIdx > Params['NChan']:
                    raise Exception("Interrupt")
            ui_updateFramePlot(chanIdx, Params, dataSet)
    except Exception as error:
        print(error)
        print('Turn off')
    