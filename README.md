# LED-board-tracking-OpenCV

This project consists of creating a OpenCV/Python server application that tracks a LED boarding (example in source), for assiting in the manufacturing process.
Besides of physically tracking the board, the system should be able to detect LEDs frequency, color, status and brightness. 
This way we eliminate the need of a human operator.

An overview of the project is illustrated in project_structure.jpg

The client connects to the server and remotely asks for information about the LED board, which is being monitored by a video processing task.
Commands avaiable to the client are to be implemented as detailled in SCPI_commands.pdf

class_model.jpg presents a general class model of the constructed solution.

The server and the video processing task should run in parallel. 

In order to track the LED board, ORB [1] features were extracted using OpenCV framework.
Furthermore a FLANN-based descriptor matcher [2] was employed.
See planne_tracker.py example of OpenCV.

The solution constructed is provided in "source"

[1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary R. Bradski in their paper ORB: An efficient alternative to SIFT or SURF, 2011

[2] Marius Muja and David G. Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", in International Conference on Computer Vision Theory and Applications (VISAPP'09), 2009 
