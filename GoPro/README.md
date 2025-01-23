This module is based on the Open GoPro tutorials that can be found on : https://gopro.github.io/OpenGoPro/tutorials/

You can either use the GoPro package directly with the code examples written below or instantiate the classes in COHN or wifi and examples can be found in the main function of the respective files.

# Wifi control
You can control the GoPro using the wifi module through command line inputs, it will first connect to the GoPro via bluetooth and then connect to the GoPro's wifi, if you are running this for the first time on a laptop you need to connect to wifi manually using the GoPro's informations written in the `GoPro/GoPro XXXX/ssid.txt` after the first run.

example:-
`python -m GoPro.wifi.WifiCommunication 'GoPro XXXX'`

You can then control the GoPro via command line, type in help to get more information on the possible controls.


# Camera On Home Network(COHN)

## 1. COHN Provision
First step, we need to instruct the GoPro to create a COHN certificate by calling the `provision_cohn.py` file. 
We need to provide the network's ssid and password through which we want to communicate and our device needs to be connected to that same network. 

An example call:-
`python -m GoPro.COHN.provision_cohn ssid password 'GoPro XXXX'`

By default it will create and write the certificate and the connection credentials in the folder `cwd/GoPro/GoPro XXXX/` where cwd is the directory you are running the code in.

For more details run :-
`python -m GoPro.COHN.provision_cohn --help`

## 2. Communicate with the GoPro via COHN
Create an instance of `COHN_communication` object by giving the path to the file `./GoPro/GoPro XXXX/credentials.txt` created before,
it will connect to the gopro via bluetooth on instantiation to be sure that the gopro is on.

example:-
`python -m GoPro.COHN.communicate_via_cohn 'GoPro/GoPro 6665/credentials.txt' `