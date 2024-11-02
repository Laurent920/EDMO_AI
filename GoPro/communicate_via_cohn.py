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

from _init_ import connect_ble, logger, ResponseManager


class COHNCommunication():
    async def __init__(self, credentials: Path) -> None:
        try:
            arguments = ''
            with open(credentials, 'r') as cred:
                arguments = cred.read() 
                
            splits = arguments.split(" ", 3)
            self.username = splits[0]
            self.password = splits[1]
            self.ip_address = splits[2]
            self.certificate =  splits[3]
            
            gopro_id = credentials.parts[-2]
            await self.gopro_ble_conenct(gopro_id)
        except Exception as e:
            logger.info(e)
            logger.info("Give the file 'credentials.txt' created by provision_cohn.py")
            
            
    async def gopro_ble_conenct(identifier):
        # Connects to the gopro via bluetooth to be sure that it's on
        manager = ResponseManager()
        try:
            await connect_ble(manager.notification_handler, identifier)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(repr(exc))
        
        
    async def send_command(self, command: str):
        match command:
            case 'start':
                op = "/gopro/camera/shutter/start"
            case 'stop':
                op = "/gopro/camera/shutter/stop"
            case 'start_stream':
                op = "/gopro/camera/stream/start"
            case 'stop_stream':
                op = "/gopro/camera/stream/stop"
            case 'get_state':
                op = "/gopro/camera/state"
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
            print('Make sure the GoPro is turned on')


async def main(cohn_com: COHNCommunication):
    while(True):
        instructions = await asyncio.gather(
                                aioconsole.ainput("Enter command: "),
                            )
        command = instructions[0]
        if command == 'quit':
            return
        await cohn_com.send_command(command)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate HTTPS communication via COHN.")
    parser.add_argument("credentials", type=str, help='Path to credentials')
    args = parser.parse_args()
    
    cohn_com = COHNCommunication(Path(args.credentials)) 
    
    asyncio.run(main(cohn_com))

