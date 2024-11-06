# communicate_via_cohn.py/Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc. (http://gopro.com/OpenGoPro).
# This copyright was auto-generated on Wed Mar 27 22:05:49 UTC 2024

from ast import arg
import sys
import json
import argparse
import asyncio
from base64 import b64encode
from pathlib import Path
import aioconsole

import requests

from GoPro import connect_ble, logger, ResponseManager


class COHNCommunication():
    def __init__(self, gopro_id: str, credentials: Path) -> None:
        try:
            self.gopro_id = gopro_id

            arguments = ''
            with open(credentials/ Path('credentials.txt'), 'r') as cred:
                arguments = cred.read() 
                
            splits = arguments.split(" ", 3)
            self.username = splits[0]
            self.password = splits[1]
            self.ip_address = splits[2]
            self.certificate = Path(splits[3]) 
        except Exception as e:
            logger.info(e)
            logger.info("Give the path to the folder created by provision_cohn.py (by default ./GoPro/GoPro XXXX/)")
            
            
    async def ble_connect(self):
        # Connects to the gopro via bluetooth to be sure that it's on
        manager = ResponseManager()
        try:
            client = await connect_ble(manager.notification_handler, self.gopro_id)
            await client.disconnect()
            logger.info(f"{self.gopro_id} ready to communicate")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(repr(exc))
        
        
    def send_command(self, command: str):
        match command:
            case 'start':
                op = "/gopro/camera/shutter/start"
            case 'stop':
                op = "/gopro/camera/shutter/stop"
            case 'start stream':
                op = "/gopro/camera/stream/start"
            case 'stop stream':
                op = "/gopro/camera/stream/stop"
            case 'get camera state':
                op = "/gopro/camera/state"
            case 'get cohn state':
                op = "/gopro/cohn/status"
            case 'get last video':
                op = "/gopro/media/last_captured"
            case _:
                print('wrong command')
                return
            
        url = f"https://{self.ip_address}" + op
        logger.debug(f"Sending:  {url}")

        token = b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"Authorization": f"Basic {token}"},
                verify=str(self.certificate),
            )
            # Check for errors (if an error is found, an exception will be raised)
            response.raise_for_status()
            logger.info("Command sent successfully")
            # Log response as json
            logger.info(f"Response: {json.dumps(response.json(), indent=4)}")
        except requests.exceptions.ConnectTimeout as timeout:
            print(f'Error: {timeout}')
            print('Make sure that the laptop is connected to the EDMO router and that the GoPro is turned on')


async def main(cohn_com: COHNCommunication):
    # await cohn_com.ble_connect()
    print('Enter quit to stop the program')
    while(True):
        instructions = await asyncio.gather(
                                aioconsole.ainput("Enter command: "),
                            )
        command = instructions[0]
        if command == 'quit':
            return
        cohn_com.send_command(command)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTPS communication via COHN.")
    parser.add_argument("credentials", type=str, help='Path to credentials')
    args = parser.parse_args()
    
    cohn_com = COHNCommunication(Path(args.credentials)) 
    
    asyncio.run(main(cohn_com))

