import numpy as np
from scipy.signal import argrelextrema # for computing the local minima

import math

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

from scipy.interpolate import interp1d

THRESHOLD = 0.15
WINDOW_SIZE = 3
MIN_TIME_OF_SIGN = 1600 # Setting the minimum time of a sign to be of 1.5 secs (1500 ms)
MIN_STOP_TIME = 1200 #350 # Minimum time seen between the stop of one sign and the start of another (500 ms)


def read_data(filename):
    data = []
    f = open(filename, "r")
    for line in f:
        line = line.replace('\n', '')
        splitted_line = line.split(',')
        splitted_line = list(map(int, splitted_line[1:])) #Antes estava splitted_line[1:]
        data.append(splitted_line)
        #for val in splitted_line:
        #    data.append(val)
    return data

def transform_data_to_sequence(data):
    time = []
    acc_x = []
    acc_y = []
    acc_z = []
    ang_x = []
    ang_y = []
    ang_z = []

    for line in data:
        time.append(line[0])
        acc_x.append(line[1])
        acc_y.append(line[2])
        acc_z.append(line[3])
        ang_x.append(line[4])
        ang_y.append(line[5])
        ang_z.append(line[6])
    return time, acc_x, acc_y, acc_z, ang_x, ang_y, ang_z

def root_mean_squared(x, y, z):
    rms = []
    for i in range(len(x)):
        rms.append(math.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
    return rms

def root_mean_squared(arr):
    rms = []
    for i in range(len(arr)):
        rms.append(math.sqrt(arr[i]**2))
    return rms 

def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
        moving_average.append(window_average)
    
    # So that the list has the same size as the data (for plotting and so on) we add <window_size - 1> more values to the end of the array
    for i in range(window_size-1):
        moving_average.append(moving_average[-1])
    return moving_average

# Returns a list with 1 in the position where it starts, -1 when it stops, and 0 if otherwise (doing the gesture or being stable)
def activation_points(time, rms_acc_x, rms_acc_y, rms_acc_z, rms_ang_x, rms_ang_y, rms_ang_z):
    activ_points = []
    summed_rms = []
    has_started = False
    last_event_time = 0
    for i in range(len(rms_acc_x)):
        val = rms_acc_x[i] + rms_acc_y[i] + rms_acc_z[i] + rms_ang_x[i] + rms_ang_y[i] + rms_ang_z[i]
        summed_rms.append(val)
    
    # Low pass filter on the Summed RMS to smooth it out
    summed_rms = moving_average(summed_rms, WINDOW_SIZE)

    for i in range(len(summed_rms)):
        if(i >= WINDOW_SIZE):
            if(has_started == False and all(summed_rms[i] > x*(1+THRESHOLD) for x in summed_rms[i-WINDOW_SIZE:i]) and time[i] > last_event_time + MIN_STOP_TIME):
                print("Started: " + str(time[i]) + " - " + str(summed_rms[i]) + " : ")
                for j in summed_rms[i-WINDOW_SIZE:i]:
                    print(str(j) + " ", end='')
                print("\n")

                has_started = True
                last_event_time = time[i]
                activ_points.append(1)
            elif(has_started == True and all(summed_rms[i] < x*(1-THRESHOLD) for x in summed_rms[i-WINDOW_SIZE:i]) and time[i] > last_event_time + MIN_TIME_OF_SIGN):
                print("Ended: " + str(time[i]) + " - " + str(summed_rms[i]) + " : ")
                for j in summed_rms[i-WINDOW_SIZE:i]:
                    print(str(j) + " ", end='')
                print("\n")
                has_started = False
                last_event_time = time[i]
                activ_points.append(-1)
            else:
                activ_points.append(0)
    
    activ_points.extend([activ_points[-1]] * WINDOW_SIZE)  # To make list the same size I append last activ_points value N times at the end (WINDOW_SIZE)
    return activ_points, summed_rms


# As the approach with the summed_rms on all channels combined didn't workout that well, now I test by computing 6 rms (each per channel)
# and computing the activation points for each of the 6 rms channels then I do an avg of the x coordinated for the 6 
def activation_points_per_channel(time, channels):
    activ_points = []
    has_started = False
    last_event_time = 0

    mvg_avg_channels = []

    for channel in channels:
        mvg_avg_channels.append(moving_average(channel, WINDOW_SIZE))

    for ch_idx, channel in enumerate(mvg_avg_channels):
        activ_points.append([])
        activ_points[ch_idx].extend([0] * WINDOW_SIZE)
        has_started = False
        last_event_time = 0
        print("Channel : " + str(ch_idx))
        for i in range(len(channel)):
            if(i >= WINDOW_SIZE):
                if(has_started == False and all(channel[i] > x*(1+THRESHOLD) for x in channel[i-WINDOW_SIZE:i]) and time[i] > last_event_time + MIN_STOP_TIME):
                    print("Started: " + str(time[i]) + " - " + str(channel[i]) + " : ")
                    for j in channel[i-WINDOW_SIZE:i]:
                        print(str(j) + " ", end='')
                    print("\n")

                    has_started = True
                    last_event_time = time[i]
                    activ_points[ch_idx].append(1)
                elif(has_started == True and all(channel[i] < x*(1-THRESHOLD) for x in channel[i-WINDOW_SIZE:i]) and time[i] > last_event_time + MIN_TIME_OF_SIGN):
                    print("Ended: " + str(time[i]) + " - " + str(channel[i]) + " : ")
                    for j in channel[i-WINDOW_SIZE:i]:
                        print(str(j) + " ", end='')
                    print("\n")
                    has_started = False
                    last_event_time = time[i]
                    activ_points[ch_idx].append(-1)
                else:
                    activ_points[ch_idx].append(0)
    
    return activ_points, mvg_avg_channels

# Uses min-max to normalize data between [0, 1]
def normalize(data):
    normalized_data = []
    min_value = min(data)
    max_value = max(data)
    for val in data:
        normalized_data.append((val - min_value)/(max_value - min_value))
    return normalized_data


# Uses the normalized data, so summed_rms always between [0, 1]
# This one yielded the best results 
# As a matter of fact, beacuse the rms is a measure of magnitude of change I can try to find the zeros of the function
def activation_points_zeros_all(time, rms_acc_x, rms_acc_y, rms_acc_z, rms_ang_x, rms_ang_y, rms_ang_z):
    activ_points = []
    summed_rms = []
    has_started = False
    last_event_time = 0

    # For now I don't count the Z as it also has the gravity present TODO: Before inserting thedata here I should remove the gravity from the values
    for i in range(len(rms_acc_x)):
        val = rms_acc_x[i] + rms_acc_y[i] + rms_ang_x[i] + rms_ang_y[i] + rms_ang_z[i]
        summed_rms.append(val)
    
    # Low pass filter on the Summed RMS to smooth it out
    summed_rms = moving_average(summed_rms, WINDOW_SIZE)
    summed_rms = normalize(summed_rms)

    for i in range(len(summed_rms)):
        if(i >= WINDOW_SIZE):
            if(has_started == False and all(x > THRESHOLD for x in summed_rms[i-WINDOW_SIZE:i]) and time[i] > last_event_time + MIN_STOP_TIME):

                has_started = True
                last_event_time = time[i]
                activ_points.append(1)
            elif(has_started == True and all(x < 1 - THRESHOLD for x in summed_rms[i-WINDOW_SIZE:i]) and time[i] > last_event_time + MIN_TIME_OF_SIGN):
                has_started = False
                last_event_time = time[i]
                activ_points.append(-1)
            else:
                activ_points.append(0)

    return activ_points, summed_rms

# Same as above, but for the two hands, the summed_rms is with the 12 channels altogether
# For the activation points I use time_l channel... but it's just as a reference, would work with the other time_r
def activation_points_zeros(time_l, time_r, summed_rms):
    activ_points = []
    has_started = False
    last_event_time = 0
    
    # Low pass filter on the Summed RMS to smooth it out
    #summed_rms = moving_average(summed_rms, WINDOW_SIZE)
    summed_rms = normalize(summed_rms)

    for i in range(len(summed_rms)):
        if(i >= WINDOW_SIZE):
            if(has_started == False and all(x > THRESHOLD for x in summed_rms[i-WINDOW_SIZE:i]) and time_l[i] > last_event_time + MIN_STOP_TIME):

                has_started = True
                last_event_time = time_l[i]
                activ_points.append(1)
            elif(has_started == True and all(x < THRESHOLD for x in summed_rms[i-WINDOW_SIZE:i]) and time_l[i] > last_event_time + MIN_TIME_OF_SIGN):
                has_started = False
                last_event_time = time_l[i]
                activ_points.append(-1)
            else:
                activ_points.append(0)

    return activ_points, summed_rms



# Receives a list with the raw_data (as returned by read_data(filename) for example) and returns a list with the timestamps of start and stop
# for now it uses the Root Mean Squared of the 6 channels which returns the magnitude of change of movement, so each sign requires a small stopagge (check/change thresholds above)
# TODO: correct the gravitational acceleration from the MPU measurements as this puts a constant change of movement on the data.
def calculate_starts_and_stops(data):
    # This separates the raw data into the time + 6 channels
    time, acc_x, acc_y, acc_z, ang_x, ang_y, ang_z = transform_data_to_sequence(data)

    # computes the root mean squared on each channel
    rms_acc_x = root_mean_squared(acc_x)
    rms_acc_y = root_mean_squared(acc_y)
    rms_acc_z = root_mean_squared(acc_z)
    rms_ang_x = root_mean_squared(ang_x)
    rms_ang_y = root_mean_squared(ang_y)
    rms_ang_z = root_mean_squared(ang_z)

    activ_points, summed_rms = activation_points_zeros_all(time, rms_acc_x, rms_acc_y, rms_acc_z, rms_ang_x, rms_ang_y, rms_ang_z)
    
    # gets the index for the starts and stops of an image
    starts = [i for i,val in enumerate(activ_points) if val==1]
    stops = [i for i,val in enumerate(activ_points) if val==-1]

    instants = []
    for i in range(len(starts)):
        instants.append(time[starts[i]])
        instants.append(time[stops[i]])

    return instants, starts, stops

# Returns a list of lists, each of the sublists is the raw_data from start to stop of a sign
def segment_data(data, starts, stops):
    signs = []
    for i in range(len(starts)):
        sign = []
        for j in range(starts[i], stops[i]):
            sign.append(data[j])
        signs.append(sign)
    return signs

def divide_file_into_separate_data():
    filename = 'words_train_50/want.txt'
    data = read_data(filename)

    instants, starts, stops = calculate_starts_and_stops(data)
    signs_raw = segment_data(data, starts, stops)

    for i in range(len(signs_raw)):
        f = open("words_train_50/want_split/want_split" + str(i) + ".txt", "w")
        sign = signs_raw[i]
        for line in sign:
            f.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "," + str(line[3]) + "," + str(line[4]) + "," + str(line[5]) + "," + str(line[6]) + "\n")
        f.close()


def calculate_starts_and_stops_2hands(data_left, data_right):
    # This separates the raw data into the time + 6 channels for each hand
    time_l, acc_x_l, acc_y_l, acc_z_l, ang_x_l, ang_y_l, ang_z_l = transform_data_to_sequence(data_left)
    time_r, acc_x_r, acc_y_r, acc_z_r, ang_x_r, ang_y_r, ang_z_r = transform_data_to_sequence(data_right)

    # computes the root mean squared on each channel
    rms_acc_x_l = moving_average(root_mean_squared(acc_x_l), WINDOW_SIZE)
    rms_acc_y_l = moving_average(root_mean_squared(acc_y_l), WINDOW_SIZE)
    rms_acc_z_l= moving_average(root_mean_squared(acc_z_l), WINDOW_SIZE)
    rms_ang_x_l = moving_average(root_mean_squared(ang_x_l), WINDOW_SIZE)
    rms_ang_y_l = moving_average(root_mean_squared(ang_y_l), WINDOW_SIZE)
    rms_ang_z_l = moving_average(root_mean_squared(ang_z_l), WINDOW_SIZE)

    rms_acc_x_r = moving_average(root_mean_squared(acc_x_r), WINDOW_SIZE)
    rms_acc_y_r = moving_average(root_mean_squared(acc_y_r), WINDOW_SIZE)
    rms_acc_z_r = moving_average(root_mean_squared(acc_z_r), WINDOW_SIZE)
    rms_ang_x_r = moving_average(root_mean_squared(ang_x_r), WINDOW_SIZE)
    rms_ang_y_r = moving_average(root_mean_squared(ang_y_r), WINDOW_SIZE)
    rms_ang_z_r = moving_average(root_mean_squared(ang_z_r), WINDOW_SIZE)

    summed_rms = []
    for i in range(len(rms_acc_x_l)):
        val = rms_acc_x_l[i] + rms_acc_y_l[i] + rms_acc_z_l[i] + rms_ang_x_l[i] + rms_ang_y_l[i] + rms_ang_z_l[i] + rms_acc_x_r[i] + rms_acc_y_r[i] + rms_acc_z_r[i] + rms_ang_x_r[i] + rms_ang_y_r[i] + rms_ang_z_r[i]
        summed_rms.append(val)

    activ_points, summed_rms = activation_points_zeros(time_l, time_r, summed_rms)
    
    # gets the index for the starts and stops of an image
    starts = [i for i,val in enumerate(activ_points) if val==1]
    stops = [i for i,val in enumerate(activ_points) if val==-1]

    return starts, stops, summed_rms, activ_points


def segment_data_2hands(data_left, data_right, starts, stops):
    signs_left = []
    signs_right = []
    for i in range(min(len(starts), len(stops))):
        sign_l = []
        sign_r = []
        for j in range(starts[i], stops[i]):
            sign_l.append(data_left[j][1:])   # Here previously it was data_left[j] so probably on main of active_segmentation I was already removing the time column
            sign_r.append(data_right[j][1:])
        signs_left.append(sign_l)
        signs_right.append(sign_r)
    return signs_left, signs_right



def tmp_function_rms_for_plots(data):
    # This separates the raw data into the time + 6 channels
    time, acc_x, acc_y, acc_z, ang_x, ang_y, ang_z = transform_data_to_sequence(data)

    # computes the root mean squared on each channel
    rms_acc_x = root_mean_squared(acc_x)
    rms_acc_y = root_mean_squared(acc_y)
    rms_acc_z = root_mean_squared(acc_z)
    rms_ang_x = root_mean_squared(ang_x)
    rms_ang_y = root_mean_squared(ang_y)
    rms_ang_z = root_mean_squared(ang_z)

    activ_points, summed_rms = activation_points_zeros_all(time, rms_acc_x, rms_acc_y, rms_acc_z, rms_ang_x, rms_ang_y, rms_ang_z)
    
    # gets the index for the starts and stops of an image
    starts = [i for i,val in enumerate(activ_points) if val==1]
    stops = [i for i,val in enumerate(activ_points) if val==-1]

    instants = []
    print(len(starts))
    print(len(stops))
    #for i in range(len(starts)):
    #    instants.append(time[starts[i]])
    #    instants.append(time[stops[i]])

    return instants, starts, stops, summed_rms, activ_points

def normalize(test_x):
    sign = []
    min_val = min(test_x)
    max_val = max(test_x)
    for val in test_x:
        sign.append((val - min_val)/(max_val - min_val))
    return sign


# https://stackoverflow.com/questions/29085268/resample-a-numpy-array
# Receives a numpy array and interpolates
def interpolate(arr, target_size):
    resample = interp1d(np.linspace(0,1, len(arr)), arr, 'linear')
    return resample(np.linspace(0,1, target_size))



if __name__ == '__main__':
    
    #USED FOR PLOTS AND TESTS FOR REPORT
    
    filename = 'words_train_50/hello.txt'
    data = read_data(filename)
    print(data)

    arr = np.array(data)
    time = arr[:, 0]
    col_x = arr[:, 1]

    mvg_avg_3 = moving_average(col_x, 5)
    mvg_avg_30 = moving_average(col_x, 30)
    """
    # Showing summed RMS example
    instants, starts, stops, summed_rms, activ_points = tmp_function_rms_for_plots(data)
    plt.plot(time, summed_rms)
    plt.title(" Summed RMS of 6 channels (1 Hand) for the Hello Sign", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("RMS", fontsize=16)
    plt.legend(['Summed RMS'], fontsize=14)
    plt.show()

    # Showing active segment areas
    plt.plot(time, summed_rms)
    plt.title(" Active Segments for the Hello Sign", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("RMS", fontsize=16)
    plt.legend(['Summed RMS', 'Areas of Active Segment'], fontsize=14)
    starts = [i for i,val in enumerate(activ_points) if val==1]
    stops = [i for i,val in enumerate(activ_points) if val==-1]
    print("Number of signs: " + str(len(starts)))
    for i in range(min(len(starts), len(stops))):
        plt.axvspan(time[starts[i]], time[stops[i]], facecolor='b', alpha=0.5)
    plt.show()


    #print(time)

    plt.plot(time, col_x, 'blue')
    plt.plot(time, mvg_avg_3, 'orange')
    plt.plot(time, mvg_avg_30, 'green')

    plt.title(" X axis acceleration of Hello Sign - Raw Data, Mvg.Avg with W=3 and W=30", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Raw Data', 'Mvg. Avg. W=3', 'Mvg. Avg. W=30'], fontsize=14) 
    plt.show()

    plt.plot(time, col_x, 'blue')

    plt.title(" X axis acceleration of Hello Sign - Raw Data", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Raw Data'], fontsize=14)
    plt.show()"""

    plt.plot(time, mvg_avg_3, 'orange')

    plt.title(" X axis acceleration of Hello Sign - Mvg.Avg with W=5", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Mvg. Avg. W=5'], fontsize=14) 
    plt.show()

    plt.plot(time, mvg_avg_30, 'green')

    plt.title(" X axis acceleration of Hello Sign - Mvg.Avg with W=30", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Mvg. Avg. W=30'], fontsize=14) 
    #plt.show()


    plt.title(" Root Mean Squared", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Mvg. Avg. W=30'], fontsize=14) 
    #plt.show()


    """ USED ALSO FOR REPORT
    filename = 'words_train_50/coffee.txt'
    data = read_data(filename)
    data = np.array(data)
    time = data[:, 0]

    instants, starts, stops, summed_rms, activ_points = calculate_starts_and_stops(data)
    signs_raw = segment_data(data, starts, stops)
    print("Number of words: " + str(len(signs_raw)))

    # Showing active segment areas
    plt.plot(time, summed_rms)
    plt.title(" Active Segments for the Want Sign", fontsize=18)
    plt.xlabel("Time [ms]", fontsize=16)
    plt.ylabel("RMS", fontsize=16)
    plt.legend(['Summed RMS', 'Areas of Active Segment'], fontsize=14)
    starts = [i for i,val in enumerate(activ_points) if val==1]
    stops = [i for i,val in enumerate(activ_points) if val==-1]
    print("Number of signs: " + str(len(starts)))
    for i in range(min(len(starts), len(stops))):
        plt.axvspan(time[starts[i]], time[stops[i]], facecolor='b', alpha=0.5)
    plt.show() """

    
    
    
    """
    plt.plot(time, summed_rms, '.')
    starts = [i for i,val in enumerate(activ_points) if val==1]
    stops = [i for i,val in enumerate(activ_points) if val==-1]
    print("Number of signs: " + str(len(starts)))
    for i in range(min(len(starts), len(stops))):
        plt.axvspan(time[starts[i]], time[stops[i]], facecolor='b', alpha=0.5)
    plt.show()
    """

    #starts, stops = act_seg.calculate_starts_and_stops_2hands(complete_data_left, complete_data_right)
    #signs_left, signs_right = act_seg.segment_data_2hands(complete_data_left, complete_data_right, starts, stops)


    #TO SPLIT TRAIN DATA AND STORE IT AND PLOT IT
    """
    filename_left = 'TrainningSet_CleanV2/GOSTO/left_hand.txt'
    filename_right = 'TrainningSet_CleanV2/GOSTO/right_hand.txt'
    data_left = read_data(filename_left)
    data_right = read_data(filename_right)
    data_left = np.array(data_left)
    data_right = np.array(data_right)
    time = data_left[:, 0]

    starts, stops, summed_rms, activ_points = calculate_starts_and_stops_2hands(data_left, data_right)
    print(str(len(starts)) + " | " + str(len(stops)))
    signs_raw_left, signs_raw_right = segment_data_2hands(data_left, data_right, starts, stops)
    plt.plot(time, summed_rms, '.')
    for i in range(min(len(starts), len(stops))):
        plt.axvspan(time[starts[i]], time[stops[i]], facecolor='b', alpha=0.5)
    plt.show()
    print("Number of words: " + str(len(signs_raw_left)) + " | " + str(len(signs_raw_right)))

    
    for i in range(len(signs_raw_left)):
        f_left = open("TrainningSet_CleanV2/GOSTO/Split/left_" + str(i) + ".txt", "w")
        f_right = open("TrainningSet_CleanV2/GOSTO/Split/right_" + str(i) + ".txt", "w")
        sign_left = signs_raw_left[i]
        sign_right = signs_raw_right[i]
        for line in sign_left:
            f_left.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "," + str(line[3]) + "," + str(line[4]) + "," + str(line[5]) + "\n")
        for line in sign_right:
            f_right.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "," + str(line[3]) + "," + str(line[4]) + "," + str(line[5]) + "\n")
        f_left.close()
        f_right.close()
        """
    
    









    """
    
    filename_left = 'TestSentences/IndividualTest/left_hand.txt'
    filename_right = 'TestSentences/IndividualTest/right_hand.txt'
    data_left = read_data(filename_left)
    data_right = read_data(filename_right)
    data_left = np.array(data_left)
    data_right = np.array(data_right)
    time = data_left[:, 0]
    acc_y = data_right[:, 2]
    acc_y = moving_average(acc_y, WINDOW_SIZE)

    starts, stops, summed_rms, activ_points = calculate_starts_and_stops_2hands(data_left, data_right)
    print(str(len(starts)) + " | " + str(len(stops)))
    signs_raw_left, signs_raw_right = segment_data_2hands(data_left, data_right, starts, stops)
    time = range(len(time))
    plt.plot(time, acc_y)
    plt.title(" Original Signal - Acceleration Y", fontsize=18)
    plt.xlabel("Samples", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Acc. Y'], fontsize=14)
    plt.show()

    time = time[starts[0]:stops[0]]
    acc_y = acc_y[starts[0]:stops[0]]
    time = range(len(time))
    print(len(time))
    plt.plot(time, acc_y)
    plt.title(" Clipped Signal - Acceleration Y", fontsize=18)
    plt.xlabel("Samples", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Acc. Y'], fontsize=14)
    plt.show()
    
    to_resize = 88
    acc_y = interpolate(acc_y, to_resize)
    acc_y = normalize(acc_y)
    time = range(to_resize)
    plt.plot(time, acc_y)
    plt.title(" Resized Signal - Acceleration Y", fontsize=18)
    plt.xlabel("Samples", fontsize=16)
    plt.ylabel("MPU Raw", fontsize=16)
    plt.legend(['Acc. Y'], fontsize=14)
    plt.show() """

    """for i in range(min(len(starts), len(stops))):
        plt.axvspan(time[starts[i]], time[stops[i]], facecolor='b', alpha=0.5)
    plt.show()
    print("Number of words: " + str(len(signs_raw_left)) + " | " + str(len(signs_raw_right))) """