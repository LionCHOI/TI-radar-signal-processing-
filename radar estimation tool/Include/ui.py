import matplotlib.pyplot as plt
import numpy as np
from keyboard import is_pressed

def plotTimeDomainData_cont(frame_idx, Params, dataSet):

    ax_list = []
    line_list = [[] for i in range(Params['NChan'])]

    currChDataQ = np.real(dataSet['rawFrameData'][0, :, :])
    currChDataI = np.imag(dataSet['rawFrameData'][0, :, :])

    plt.ion()
    fig = plt.figure(figsize=(10, 10), layout="constrained")
    fig.suptitle("Frame {}".format(frame_idx), fontsize=20)

    for chanIdx in range(Params['NChan']):
        ax_list.append(plt.subplot(2, 2, chanIdx + 1))  # subplot start Index is 1
        line_list[chanIdx].append(ax_list[chanIdx].plot(currChDataQ[chanIdx,:], label='I (In-phase)', color='orange')[0])    # Returns a tuple of line objects, thus the comma
        line_list[chanIdx].append(ax_list[chanIdx].plot(currChDataI[chanIdx,:], label='Q (Quadrature)', color='blue')[0])
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title(f'Channel {chanIdx + 1}')
        # plt.legend()
        plt.grid(True)

    try:
        print("press 'ESC' to terminate")
        for data_chrip in dataSet['rawFrameData'][:]:
            # generate data 
            currChDataQ = np.real(data_chrip)
            currChDataI = np.imag(data_chrip)

            for chanIdx in range(Params['NChan']):    
                line_list[chanIdx][0].set_ydata(currChDataQ[chanIdx, :])
                line_list[chanIdx][1].set_ydata(currChDataI[chanIdx, :])    ## return 값을 다르게 설정하면 된다.

            fig.canvas.draw()
            fig.canvas.flush_events()

            if is_pressed('esc'):
                raise Exception('interrupt by pressing "ESC"')
    except Exception as error:
        print(error)
    finally:
        print('Turn off')
        plt.close()     
          

def plot1DRangeProfile_cont(frame_idx, Params, dataSet, linearMode):

    # variables
    line_list = [i for i in range(Params['NChan'])]
    rangeProfileData = dataSet['radarCubeData'][:, :, :]
    rangeBin = dataSet['rangeBin']  # range scope

    if linearMode == 1:            
        rangeProfileData = np.abs(rangeProfileData[:])
    else:
        rangeProfileData[rangeProfileData[:] == 0] = 1  ## log 0 is error
        rangeProfileData = 20 * np.log10(np.abs(rangeProfileData[:]))

    plt.ion()
    fig = plt.figure(figsize=(10, 10), layout="constrained" )
    fig.suptitle("Frame {}".format(frame_idx), fontsize=20)

    for chanIdx in range(Params['NChan']):
        ax = plt.subplot(2, 2, chanIdx + 1)  # subplot start Index is 1
        line_list[chanIdx] = ax.plot(rangeBin, rangeProfileData[0, chanIdx])[0]
        plt.title(f'1D Range Profile - Channel {chanIdx + 1}')
        plt.xlabel('Range Bins')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True)

    try:
        print("press 'ESC' to terminate")
        for rangeProfileData_chrip in rangeProfileData[:]: 
            for chanIdx in range(Params['NChan']):    
                line_list[chanIdx].set_ydata(rangeProfileData_chrip[chanIdx])

            fig.canvas.draw()
            fig.canvas.flush_events()
    
            if is_pressed('esc'):
                raise Exception('interrupt by pressing "ESC"')
    except Exception as error:
        print(error)
    finally:
        print('Turn off')
        plt.close()
        
        
def plot_all(chanIdx, Params, dataSet, linearMode, all):
    
    FIRST_CHIRP, FIRST_FRAME = 0, 0
    ROW_COLUMN_SET = 1
    THRESHLOD = 1e4
    
    if all == False:
        plt_list = [0, 0]
    else:
        plt_list = [0, 0, 0, []]
        
    plt_num = len(plt_list)
    
    # dataSet['DATA'] = np.ndarray(['NFrame']['NChan']['NChirp']['NSample'])
    currChDataQ = np.real(dataSet['rawFrameData_set'][:, chanIdx])
    currChDataI = np.imag(dataSet['rawFrameData_set'][:, chanIdx])
    Range_fft = np.abs(dataSet['radarCubeData_1D_FFT_set'][:, chanIdx])
    Range_Doppler_fft = np.abs(dataSet['radarCubeData_2D_FFT_set'][:, chanIdx])

    # dataSet['radarCubeData_Bartlett_set'] = np.ndarray(['NFrame']['NSample']['angle_res_size'])
    angle_result = np.abs(dataSet['radarCubeData_Bartlett_set'])

    if linearMode == False:            
        linear_func = lambda x: 20 * np.log10(x)
        THRESHLOD, Range_fft, Range_Doppler_fft = list(map(linear_func, [THRESHLOD, Range_fft, Range_Doppler_fft]))

    pt_range = dataSet['rangeBin'].flatten()
    pt_velocity = dataSet['dopplerBin'].flatten()
    angle_x, angle_y = Params['angle']['x'], Params['angle']['y']
    
    
    plt.ion()
    fig = plt.figure("Channel {}".format(chanIdx+1), figsize=(10, 5), layout="constrained")    # The larger figsize, The slower speed & speed is slower when set the layout 
    # fig = plt.figure("Channel {}".format(chanIdx+1), figsize=(10, 5))   
    
    ## initialize
    for Idx in range(plt_num):
        if all == False:
            ax_tmp = plt.subplot(ROW_COLUMN_SET, plt_num, Idx + 1)  # subplot start Index is 1
        else:
            ax_tmp = plt.subplot(plt_num//2, plt_num//2, Idx + 1)  # subplot start Index is 1
        if Idx == 0:
            print(angle_x.shape, angle_y.shape, angle_result.shape)
            print(pt_velocity.shape, pt_range.shape, Range_Doppler_fft.shape)
            
            plt_list[Idx] = ax_tmp.pcolormesh(angle_x, angle_y, angle_result[FIRST_FRAME], vmax = angle_result[FIRST_FRAME].max(), shading='auto')
            plt.xlabel('x')
            plt.ylabel('y')
            fig.colorbar(plt_list[Idx], ax=ax_tmp, aspect=10)
        elif Idx == 1:
            plt_list[Idx] = ax_tmp.pcolormesh(pt_velocity, pt_range, Range_Doppler_fft[FIRST_FRAME].T, vmax = Range_Doppler_fft[FIRST_FRAME].max(), shading='nearest')
            plt.xlabel('velocity')
            plt.ylabel('range')
            fig.colorbar(plt_list[Idx], ax=ax_tmp, aspect=10)
        elif Idx == 2:
            plt_list[Idx] = ax_tmp.plot(pt_range, Range_fft[FIRST_FRAME, FIRST_CHIRP])[0]
            # for sampleIdx in range(Params['numRangeBins']):
            #     if Range_fft[FIRST_FRAME, FIRST_CHIRP, sampleIdx] > THRESHLOD:
            #         plt.text(pt_range[sampleIdx], Range_fft[FIRST_FRAME, FIRST_CHIRP, sampleIdx], '{:.2f}'.format(pt_range[sampleIdx]))
            plt.xlabel('Range Bins')
            if linearMode == True:
                plt.ylabel('Amplitude')
            else:
                plt.ylabel('Amplitude (dB)')
        else:
            plt_list[Idx].append(ax_tmp.plot(currChDataQ[FIRST_FRAME, FIRST_CHIRP], label='I (In-phase)', color='orange')[0]) 
            plt_list[Idx].append(ax_tmp.plot(currChDataI[FIRST_FRAME, FIRST_CHIRP], label='Q (Quadrature)', color='blue')[0]) 
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')

    ## Keep generating
    try:
        print("press 'ESC' to terminate")
        for frameIdx in range(Params['NFrame']):
            print(frameIdx)
            
            for Idx in range(plt_num):
                if Idx == 0:
                    plt_list[Idx].set_array(angle_result[frameIdx])
                elif Idx == 1:
                    plt_list[Idx].set_array(Range_Doppler_fft[frameIdx].T)
                elif Idx == 2:
                    plt_list[Idx].set_ydata(Range_fft[frameIdx, FIRST_CHIRP])
                    # for sampleIdx in range(Params['numRangeBins']):
                    #     if Range_fft[FIRST_FRAME, FIRST_CHIRP, sampleIdx] > THRESHLOD:
                    #         plt.text(pt_range[sampleIdx], Range_fft[FIRST_FRAME, FIRST_CHIRP, sampleIdx], '{:.2f}'.format(pt_range[sampleIdx]))
                else:
                    plt_list[Idx][0].set_ydata(currChDataQ[frameIdx, FIRST_CHIRP])
                    plt_list[Idx][1].set_ydata(currChDataI[frameIdx, FIRST_CHIRP])    ## return 값을 다르게 설정하면 된다.
                
            fig.canvas.draw()
            fig.canvas.flush_events()
    
            if is_pressed('esc'):
                raise Exception('interrupt by pressing "ESC"')
    except Exception as error:
        print(error)
    finally:
        print('Turn off')
        plt.close()

####################################################################################################################################



def ui_updateFramePlot(chanIdx, Params, dataSet):    

    LINEAR_MODE, ALL = True, False  # If Linear mode is False, DB mode 

    # plot1DRangeProfile_cont(frame_idx, Params, dataSet, linearMode = False)
    # plotTimeDomainData_cont(frame_idx, Params, dataSet)
    
    plot_all(chanIdx, Params, dataSet, LINEAR_MODE, ALL)

if __name__ == "__main__":
    print(":)")