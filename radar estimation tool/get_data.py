from Include.load import *
from Include.generating_2 import *
from Include.params import *
# from Include.validation import *
# from Include.export import *
from Include.ui import *
from Include.angle import *

## variables
SETUP_JSON_FILE_NAME = '22xx.setup.json'
RAW_DATA_FILE_NAME = 'mydata.mat'
RADAR_CUBE_DATA_FILE_NAME ='radarcube.mat'
DEBUG_PLOT = True # True = plot, False = no plot
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
radarCube_init(Params, dataSet, CFAR, DEBUG_PLOT) # set the paramter for estimation

dataSet['rawDataComplex_set'] = dp_loadFrameData(Params['fid_rawData'][0], Params['dataSizeOneFrame'])
dataSet['rawFrameData_set'] = np.zeros((Params['NFrame'], Params['NChan'], Params['NChirp'], Params['NSample']), dtype='complex')
dataSet['radarCubeData_1D_FFT_set'] = np.zeros((Params['NFrame'], Params['NChan'], Params['NChirp'], Params['NSample']), dtype='complex')
dataSet['radarCubeData_2D_FFT_set'] = np.zeros((Params['NFrame'], Params['NChan'], Params['NChirp'], Params['NSample']), dtype='complex')

if DEBUG_PLOT is True:
    Params['plot'] = plot_init(Params, dataSet) # plot

# processing
NUM_FRAME = 100
NUM_TARGET = 1
COV_MAT = np.zeros((NUM_FRAME, 4, 4)).astype('complex_')
ANGLE_VALUES = np.zeros((NUM_FRAME, NUM_TARGET))
for frame_Idx in range(Params['NFrame']):
    dp_updateFrameData(frame_Idx, Params, dataSet, COV_MAT, ANGLE_VALUES)