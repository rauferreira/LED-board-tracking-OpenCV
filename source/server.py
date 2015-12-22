'''
server
======

Structure of DATA from IP task:


----------------------------------------

'''

import socket
import sys
from thread import *



def send_data(data):
    '''gets data from IP task'''
    global DATA
    DATA= data
    parse_data(DATA)


 
def start_server():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('127.0.0.1', 8606)
    print "Starting server on", server_address, "..."

    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(100)

    while True:
    #Accepting incoming connections
        conn, addr = sock.accept()
    #Creating new thread. Calling clientthread function for this function and passing conn as argument.
        start_new_thread(talk_to_client,(conn,)) #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
 
    conn.close()
    sock.close()


def parse_data(data):
    global NAME
    global STATUS
    global COLOR_NAME
    global COLOR_RGB
    global FREQUENCY
    global BRIGHTNESS
    global FPS
    global IMAGE
    
    NAME, STATUS, COLOR_NAME, COLOR_RGB, FREQUENCY, BRIGHTNESS, FP, IMAGE = DATA

def parse_command(command):
    a= command.split(":")
    print a
    if len(a)== 2 or len(a)== 3:
        return a
    else:
        return None
    
def LED_with_3(values):
    '''command about LED properties with three parameters'''
    if values[2]== "status":
        sta= STATUS[NAME.index(values[1])]
        if sta== True:
            return "On"
        else:
            return "Off"
    if values[2]== "color_name":
        return COLOR_NAME[NAME.index(values[1])]
    if values[2]== "color_rgb":
        rgb= COLOR_RGB[NAME.index(values[1])]
        return "R: "+ str(rgb[2])+ " G: "+ str(rgb[1])+ " B: "+ str(rgb[0])
    if values[2]== "frequency":
        if FREQUENCY[NAME.index(values[1])]== -1:
            return "The requensted LED hasn't switched on yet!"
        elif FREQUENCY[NAME.index(values[1])]== -2:
            return "The requested LED hasn't switched off yet!"
        else:
            return "{0:.2f}".format(FREQUENCY[NAME.index(values[1])])+ " Hz"
    if values[2]== "brightness":
        if BRIGHTNESS[NAME.index(values[1])]== -1:
            return "The requensted LED hasn't switched on yet!"
        else:
            return "{0:.2f}".format(BRIGHTNESS[NAME.index(values[1])])

def LED_with_2(values):
    '''command about LED properties with two parameters'''
    if values[1]== "numbof":
        return str(len(NAME))+" LEDs"
    else:
        return ", ".join(NAME)

def IMAGE_with_3(values):
    '''command about IMAGE properties with three parameters'''
    return 0

def IMAGE_with_2(values):
    '''command about IMAGE properties with two parameters'''
    if values[1]== "fps":
        return str(FPS)+ " fps"

def talk_to_client(connection):
    try:
        command = connection.recv(128)
        
        if len(command)!= 0:
            print
            print "Received command:", command
            
            # if 'close all' is received stop the server and exit
            if command== "close all":
                print "Closing connection and stopping server..."
                connection.close()
                sock.close()
                return 0

            else:
                values= parse_command(command)
                # if not properly formatted, return 'invalid command' to client
                if values== None:
                    print("Invalid command! Please check your command.")
                    connection.sendall("Invalid command! Please check your command.")
                else:
                    # if the command is about LED properties
                    if values[0]== "LED":
                        if len(values)== 3:
                            if values[1] not in NAME:
                                print("Please check the name of LED you have entered.")
                                connection.sendall("Please check the name of LED you have entered.")
                            elif values[2] not in ["status", "color_name", "color_rgb", "frequency", "brightness"]:
                                print("Please check the property of LED you have entered.")
                                connection.sendall("Please check the property of LED you have entered.")
                            else:
                                answer= LED_with_3(values)
                                if answer== None:
                                    answer= "Haven't calculated the requested answer yet!"
                                print "Response to client:", answer
                                try:
                                    connection.sendall(answer)
                                except Exception, e:
                                    print e
                                    print "Error: ", sys.exc_info()[0]
                                    connection.sendall("Error sending the answer from server to client!")

                        elif len(values)== 2:
                            if values[1] not in ["numbof", "list"]:
                                print("Please check the property of LED_LIST you have entered.")
                                connection.sendall("Please check the property of LED_LIST you have entered.")
                            else:
                                answer= LED_with_2(values)
                                print "Response to client:", answer
                                try:
                                    connection.sendall(answer)
                                except Exception, e:
                                    print e
                                    print "Error: ", sys.exc_info()[0]
                                    connection.sendall("Error sending the answer from server to client!")
                        else:
                            print("Invalid number of LED specifications.")
                            connection.sendall("Invalid number of LED specifications.")
                    
                    # if the command is about IMAGE properties
                    elif values[0]== "IMAGE":
                        if len(values)== 3:
                            pass
                        elif len(values)== 2:
                            if values[1] not in ["fps"]:
                                print("Please check the property of IMAGE you have entered.")
                                connection.sendall("Please check the property of IMAGE you have entered.")
                            else:
                                answer= IMAGE_with_2(values)
                                print "Response to client:", answer
                                try:
                                    connection.sendall(answer)
                                except Exception, e:
                                    print e
                                    print "Error: ", sys.exc_info()[0]
                                    connection.sendall("Error sending the answer from server to client!")
                        else:
                            print("Invalid number of IMAGE specifications.")
                            connection.sendall("Invalid number of IMAGE specifications.")

                    # ifcommand doesn't belong to any of these categories
                    elif values[0]== "PATH":
                        print "PATH to config file: " + values[1]
                    elif values[0]== "SAVE":
                        cv2.imwrite(values[1], IMAGE, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    else:
                        print("Please check the Element_Type you have entered.")
                        connection.sendall("Please check the Element_Type you have entered.")
                    
                    
        else:
            ## if client got disconnected from server 
            ## press CTRL+c in client to disconnect!
            #print "Client disconnected the connection. Stopping the server..."
            #connection.close()
            #sock.close()
            return 0         
    except Exception, e:
        print e
        print "Error: ", sys.exc_info()[0]
        print "Error with server!"
        sys.exit()


def run():
    print __doc__
    sock, connection= start_server()


#main()
        