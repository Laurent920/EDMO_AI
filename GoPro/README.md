# Camera On Home Network(COHN)

## 1. COHN Provision
First step, we need to instruct the GoPro to create a COHN certificate by calling the `provision_cohn.py` file. 
We need to provide the network's ssid and password. 

An example call:-
`python provision_cohn.py ssid password -i 'GoPro XXXX'`

By default it will create and write the certificate and the connection credentials in the folder `/GoPro XXXX/`

For more details run :-
`python provision_cohn.py --help`

## 2. Communicate with the GoPro via COHN
Create an instance of `COHN_communication` object by giving the file `credentials.txt` created before:-
`python communicate_via_cohn.py "GoPro 6665/credentials.txt" `