#import FBIB_LIBS.readArduino as rA
#import FBIB_LIBS.calibration as cal
#import FBIB_LIBS.training as tr
#import FBIB_LIBS.filter as filter
#import FBIB_LIBS.classifier as classifier
#import FBIB_LIBS.powerSpectrumDensity as psd

import word2speech as w2s
#import classifier
import active_segmentation as act_seg
import decision_tree as dt

import openai
import serial
import time
import numpy as np
import scipy as sc
import pyttsx3
from gtts import gTTS
import os  


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

import socket # Package for the internet stuff

# The IP that is printed in the serial monitor from the ESP32 when you connect it
UDP_IP_RIGHT_HAND = "192.168.1.197" #"192.168.1.197" #"192.168.1.83"
UDP_IP_LEFT_HAND = "192.168.1.198" #"192.168.1.198" #"192.168.1.89" 
#BROADCAST_IP = "192.168.1.255"
SHARED_UDP_PORT_RIGHT = 4210
SHARED_UDP_PORT_LEFT = 4211

MAX_NUMBER_MSGS = 100

# Does some msg exchanges to finally receive the SeeedUino's IP
def receiveSeedUinoIp():
    # First it broadcast a msg to all network IPs (eventually the seedUino)
    broadcastIp = "192.168.1."
    for i in range(256):
        broadcastIp_tmp = broadcastIp + str(i)
        broadcast = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
        broadcast.connect((broadcastIp_tmp, SHARED_UDP_PORT))
        broadcast.send('Hello ESP32'.encode())
        broadcast.shutdown(socket.SHUT_RDWR)
        broadcast.close()
    # now the SeeedUino knows my computer's IP so I will wait until receiving a msg from him
    rec = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
    rec.connect(('127.0.0.1', SHARED_UDP_PORT)) # Here I define the localhost IP (default)
    readMsg = rec.recv(2048)
    return readMsg

def establishConnection(conn):
    conn.send('Hello ESP32'.encode())
    readMsg = conn.recv(2048)
    return

def read_data(msg_left_hand, msg_right_hand, msgs_list, msg_counter):
    if msg_counter == MAX_NUMBER_MSGS:
        msg_counter = 0
    msg_left_hand = msg_left_hand.split()
    msg_right_hand = msg_right_hand.split()
    msgs_list[0].insert(msg_counter, msg_right_hand)
    msgs_list[1].insert(msg_counter, msg_left_hand)
    msg_counter = msg_counter + 1
    
    return msgs_list, msg_counter

def read_one_hand_data(msg_hand, msgs_list, msg_counter):
    if msg_counter == MAX_NUMBER_MSGS:
        msg_counter = 0
    msg_hand = msg_hand.split()
    msgs_list.insert(msg_counter, msg_hand)
    return msgs_list, msg_counter

def textToSpeech(phrase, language = 'pt'):
    myobj = gTTS(text=phrase, lang=language, slow=False)
    myobj.save("to_speak.mp3")
    os.system("to_speak.mp3")


complete_data_left = []
complete_data_right = []


#fp_left = open("TrainningSet_3Words/Phrase/left_hand.txt", "w")
#fp_right = open("TrainningSet_3Words/Phrase/right_hand.txt", "w")


if __name__ == '__main__':
    
    
    # Set up connection to Arduino
    left_hand = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
    left_hand.connect((UDP_IP_LEFT_HAND, SHARED_UDP_PORT_LEFT))
    establishConnection(left_hand)

    right_hand = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
    right_hand.connect((UDP_IP_RIGHT_HAND, SHARED_UDP_PORT_RIGHT))
    establishConnection(right_hand)

    print("Starting\n")
    try:
        while(True):
            raw_data_left = (left_hand.recv(2048)).decode("utf-8")
            raw_data_right = (right_hand.recv(2048)).decode("utf-8")

            #print(raw_data_left)
            #print(raw_data_right)

            #fp_left.write(raw_data_left + "\n")
            #fp_right.write(raw_data_right + "\n")

        
            splitted_data_left = raw_data_left.split(',')
            splitted_data_left = list(map(int, splitted_data_left[1:]))

            splitted_data_right = raw_data_right.split(',')
            splitted_data_right = list(map(int, splitted_data_right[1:]))
            
            complete_data_left.append(splitted_data_left)
            complete_data_right.append(splitted_data_right) 
            

    except KeyboardInterrupt:
        print("Thank you :)")

    

    #fp_left.close()
    #fp_right.close() 

    # Now that we have a complete phrase in raw data we can process it
    # First we join the two sequences, creating a matrix with 12 columns (6 on left for left hand and 6 on the right for right hand)
    
    #filename_left = 'TrainningSet_3Words/Phrase/left_hand.txt'
    #filename_right = 'TrainningSet_3Words/Phrase/right_hand.txt'
    #complete_data_left = act_seg.read_data(filename_left)
    #complete_data_right = act_seg.read_data(filename_right)

    starts, stops, summed_rms, activ_points = act_seg.calculate_starts_and_stops_2hands(complete_data_left, complete_data_right)
    signs_left, signs_right = act_seg.segment_data_2hands(complete_data_left, complete_data_right, starts, stops)

    """tmp_data = np.array(complete_data_left)
    time = tmp_data[:, 0]
    plt.plot(time, summed_rms, '.')
    for i in range(min(len(starts), len(stops))):
        plt.axvspan(time[starts[i]], time[stops[i]], facecolor='b', alpha=0.5)
    plt.show() """


    #print(signs_left)
    classified_signs = []
    
    # TODO: complete decision_tree.py with the functions to be imported here (right now they are in the main of the decision_tree.py)
    #model = dt.load_model("decision_tree_model") 
    model = dt.load_model("decision_tree_clean_3words") 
    for sign_idx in range(len(signs_left)):
        predict = dt.predict(model, signs_left[sign_idx], signs_right[sign_idx], dt.MAX_SAMPLE_SIZE)
        classified_signs.append(predict[0])

    print(classified_signs)
    
    openai.api_key = w2s.getKey()
    phrase = w2s.makeFluid_2hands(openai, classified_signs)
    print(phrase)

    # Speak phrase out loud 
    textToSpeech(phrase)
    

    right_hand.shutdown(socket.SHUT_RDWR)
    right_hand.close()
    
    left_hand.shutdown(socket.SHUT_RDWR)
    left_hand.close()


