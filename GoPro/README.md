This module is based on the Open GoPro tutorials that can be found on : https://gopro.github.io/OpenGoPro/tutorials/

# Camera On Home Network(COHN)

## 1. COHN Provision
First step, we need to instruct the GoPro to create a COHN certificate by calling the `provision_cohn.py` file. 
We need to provide the network's ssid and password. 

An example call:-
`python -m GoPro.COHN.provision_cohn ssid password 'GoPro XXXX'`

By default it will create and write the certificate and the connection credentials in the folder `cwd/GoPro/GoPro XXXX/` where cwd is the directory you are running the code in.

For more details run :-
`python -m GoPro.COHN.provision_cohn --help`

## 2. Communicate with the GoPro via COHN
Create an instance of `COHN_communication` object by giving the path to the file `./GoPro/GoPro XXXX/credentials.txt` created before,
it will connect to the gopro via bluetooth on instantiation to be sure that the gopro is on.

example:-
`python -m GoPro.COHN.communicate_via_cohn 'GoPro/GoPro 6665/credentials.txt'" `