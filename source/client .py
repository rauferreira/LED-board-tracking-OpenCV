'''
client
======

Recognized commands:
    close all - stops server and client. Press Esc to close the window
           V1   V2
    LED:<name>:status - returns the status of the selected LED; On/Off
    LED:<name>:color_name - return color name of the selected LED; red/yellow/green/cyan/blue
    LED:<name>:color_rgb - returns color RGB values; R: 150 G: 60 B:190
    LED:<name>:freqyency - returns blinking frequency of the selected LED; 1.2 Hz
    LED:<name>:brightness - returns brightness of the led area



    LED:numbof - returns total number of LEDs; 5 LEDs
    LED:list - returns names of all the marked LEDs

    IMAGE:fps - returns frames per second of the current video; 27 fps

    IMAGE:store:int -
    IMAGE:store:ext -

----------------------------------------

'''

import socket
import sys


def start_client():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('127.0.0.1', 8606)
    print "Connecting to", server_address, "..."

    try:
        sock.connect(server_address)
        print "Sucessfully connected to server!"
    except:
        print "Unable to connect to the server!"
        sys.exit()

    return sock

def start_client2():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('127.0.0.1', 8607)
    print "Connecting to", server_address, "..."

    try:
        sock.connect(server_address)
        print "Sucessfully connected to server!"
    except:
        print "Unable to connect to the server!"
        sys.exit()

    return sock

def talk_to_server(sock, cmd):
    '''
    sends commands to server and displays response on the terminal
    '''
    #print "\nEnter your command:",
    #user_command= raw_input()
    user_command = cmd
    print "COMMAND REQUESTED: " + str(user_command)
    try:
        # Send command
        message = user_command
        sock.sendall(message)

        if message== "close all":
            print "Closing connection and stopping server!"
            sock.close()
            return 0

        # receive output
        response= sock.recv(128)
        print "Response from server:", response
        sock.close()
        sys.exit()
    except:
        print(sys.stderr)
        sock.close()
        sys.exit()

def send_path_to_server(sock, path):
    try:

        sock.sendall(path)
        return 0
    except:
        print "Error has occured! (client)"
        print(sys.stderr)
        sock.close()
        sys.exit()


def main():
    print __doc__

    

    if "PATH" in sys.argv[1]:
    	sock= start_client2()
        while True:
            ret= send_path_to_server(sock, sys.argv[1])
            if ret== 0:
                break

    else:
    	sock= start_client()
        while True:
            ret= talk_to_server(sock,sys.argv[1])
            if ret== 0:
                break

main()
