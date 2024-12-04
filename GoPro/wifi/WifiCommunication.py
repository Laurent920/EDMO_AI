# wifi_enable.py/Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc. (http://gopro.com/OpenGoPro).
# This copyright was auto-generated on Wed, Sep  1, 2021  5:06:01 PM

import json
import asyncio
import argparse
from pathlib import Path
import os
import re
import requests
import aioconsole
import subprocess
import time

from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic

from GoPro import logger, GOPRO_BASE_URL, connect_ble, GoProUuid


class WifiCommunication():
    def __init__(self, gopro_id: str, folder:Path = None) -> None:
        pattern = r"GoPro \d{4}"
        if re.match(pattern, gopro_id):
            self.gopro_id = gopro_id
        else:
            raise ValueError("GoPro id must be of the format 'GoPro XXXX'")
        
        self.saveFile = Path(os.getcwd())
        self.saveFile /= folder if folder else Path(f'GoPro/{self.gopro_id}')
        self.saveFile /= Path('ssid.txt')
        
        self.ssid = None
        self.password = None
        if folder:
            try:
                arguments = ''
                with open(folder / Path('ssid.txt'), 'r') as cred:
                    arguments = cred.read() 
                    
                splits = arguments.split(" ", 2)
                self.ssid = splits[0]
                self.password = splits[1]
                
                self.gopro_id = folder.parts[-1]
            except Exception as e:
                logger.info(e)
                logger.info("Give the path to the folder ./GoPro/GoPro XXXX/ where ssid.txt is.")
            
        self.initialized = False
        
        
    async def initialize(self):
        await self.enable_wifi(self.ssid, self.password)
        
        self.connect_wifi()
        
        self.initialized = True
    

    async def enable_wifi(self, ssid: str|None = None, password: str|None = None) -> None:
        """Connect to a GoPro via BLE, find its WiFi AP SSID and password if not given, and enable its WiFI AP"""
        
        # Synchronization event to wait until notification response is received
        event = asyncio.Event()
        client: BleakClient

        async def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray) -> None:
            uuid = GoProUuid(client.services.characteristics[characteristic.handle].uuid)
            logger.info(f'Received response at {uuid}: {data.hex(":")}')

            # If this is the correct handle and the status is success, the command was a success
            if uuid is GoProUuid.COMMAND_RSP_UUID and data[2] == 0x00:
                logger.info("Command sent successfully")
            # Anything else is unexpected. This shouldn't happen
            else:
                logger.error("Unexpected response")

            # Notify the writer
            event.set()

        client = await connect_ble(notification_handler, self.gopro_id)

        if not ssid or not password:
            # Read from WiFi AP SSID BleUUID
            ssid_uuid = GoProUuid.WIFI_AP_SSID_UUID
            logger.info(f"Reading the WiFi AP SSID at {ssid_uuid}")
            ssid = (await client.read_gatt_char(ssid_uuid.value)).decode()
            self.ssid = ssid
            logger.info(f"SSID is {ssid}")

            # Read from WiFi AP Password BleUUID
            password_uuid = GoProUuid.WIFI_AP_PASSWORD_UUID
            logger.info(f"Reading the WiFi AP password at {password_uuid}")
            password = (await client.read_gatt_char(password_uuid.value)).decode()
            self.password = password
            logger.info(f"Password is {password}")
            
            logger.info(f"Saving login data in {self.saveFile}")
            with open(self.saveFile, 'w') as f:
                f.write(f'{self.ssid} {self.password}')
            

        # Write to the Command Request BleUUID to enable WiFi
        logger.info("Enabling the WiFi AP")
        event.clear()
        request = bytes([0x03, 0x17, 0x01, 0x01])
        command_request_uuid = GoProUuid.COMMAND_REQ_UUID
        logger.debug(f"Writing to {command_request_uuid}: {request.hex(':')}")
        await client.write_gatt_char(command_request_uuid.value, request, response=True)
        await event.wait()  # Wait to receive the notification response
        logger.info("WiFi AP is enabled")
        
        await client.disconnect()

        
    def connect_wifi(self):
        for i in range(5):
            # Attempt to connect
            logger.info(f"Connecting to the gopro wifi: {self.ssid}")   
            response = subprocess.run(f'netsh wlan connect name="{self.ssid}"', capture_output=True, text=True, shell=True)
            if response.returncode != 0:
                logger.info(f"Connection failed: {response.stderr}")
                time.sleep(5)
            
            # Confirm connection by checking the network status
            time.sleep(5)
            try:
                if self.verify_wifi_connection():
                    return 
            except ValueError:
                pass
            if i == 0:
                logger.info(f"""For first time connection on a device please connect to gopro's wifi manually using :
                            ssid: {self.ssid}, password: {self.password}""")
            logger.info(f"Reattempting to connect {i+1}/5")       
        raise ValueError("Unable to connect to the gopro's wifi")
            
    
    def verify_wifi_connection(self):
        check_response = subprocess.run("netsh wlan show interfaces", capture_output=True, text=True, shell=True)
        if self.ssid in check_response.stdout:
            logger.info('Succesfully connected to the wifi')
            return True
        else:
            raise ValueError("Unable to connect to the gopro's wifi")
        
        
    def send_command(self, command:str, savePath:str = None):
        if not self.initialized:
            # self.verify_wifi_connection()
            self.initialized = True
            
        download_video = False
        file = ''
        querystring = {}
        match command:
            case 'camera control':
                op = "/gopro/camera/control/set_ui_controller"          
                querystring = {"p":"0"}
            case 'get preset':
                op = "/gopro/camera/presets/get"
            case 'set video':
                op = '/gopro/camera/presets/set_group?id=1000'
            case 'set photo':
                op = "/gopro/camera/presets/set_group?id=1001" 
            case 'load preset':
                op = "/gopro/camera/presets/load"
                querystring = {"id":"6"}
            case 'keep alive':
                # Good practice: send every 3 seconds
                op = "/gopro/camera/keep_alive"
            case 'start':
                op = "/gopro/camera/shutter/start"
            case 'stop':
                op = "/gopro/camera/shutter/stop"
            case 'start stream':
                op = "/gopro/camera/stream/start"
                print('View stream in VLC via Media -> Open Network Stream : udp://@:8554')
            case 'stop stream':
                op = "/gopro/camera/stream/stop"
            case 'get camera state':
                op = "/gopro/camera/state"
            case 'get cohn state':
                op = "/gopro/cohn/status"
            case 'get media list':
                op = "/gopro/media/list"
            case 'turbo mode':
                op = '/gopro/media/turbo_transfer'
                querystring = {"p":"0"}
            case 'get last video':
                op = "/gopro/media/last_captured"
                download_video = True
            case 'get video':
                file = input('Enter the name of the file to download from the GoPro: ')
                op = f"/videos/DCIM/100GOPRO/{file}"
                download_video = True
            case 'hilight':    
                op = "/gopro/media/hilight/moment"
            case 'help':
                # TODO 
                print('options details need to be added')
                return
            case _:
                print('wrong command')
                return
        
        # op = "/gopro/camera/presets/set_group?id=1000"
        url = GOPRO_BASE_URL + op
        logger.info(f"Sending {url}")

        # Send the GET request and retrieve the response
        response = requests.get(url, timeout=10, params=querystring)
        # Check for errors (if an error is found, an exception will be raised)
        response.raise_for_status()
        logger.info("Command sent successfully")
        
        logger.info(response)
        if not download_video:
            logger.info(logger.info(f"Response: {json.dumps(response.json(), indent=4)}"))
        else:
            if not savePath:
                logger.error("savePath argument is missing in send_command ==> saving in current directory...")
                savePath = os.getcwd()
            
            try:
                rsp = response.json()
                logger.info(rsp)
                for i in range(3):
                    if 'file' in rsp.keys():
                        file = rsp['file']
                        url = GOPRO_BASE_URL + f"/videos/DCIM/100GOPRO/{file}"
                        response = requests.get(url, timeout=10)            
                        logger.info(response)
                        time.sleep(3)
                    if response.status_code == 200:
                        break
                    else:
                        print(f"Error response code, failed to retrieve the video retrying... {i+1}/3")
            except:
                pass
            savePath = Path(savePath) / Path(file)
            logger.info(f"Saving video {file} at {savePath}")
            open(savePath, 'wb').write(response.content)


async def main(wifi_com: WifiCommunication):
    await wifi_com.initialize()
    print('Enter quit to stop the program')
    while(True):
        instructions = await asyncio.gather(
                                aioconsole.ainput("Enter command (separate command and save path with '-' to download file to specific place): "),
                            )
        command:str = instructions[0]
        if command == 'quit':
            return
        splits:str = command.split('-', 2)
        path = None if len(splits) < 2 else splits[1]
        wifi_com.send_command(splits[0], path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Connect to a GoPro camera via BLE, get its WiFi Access Point (AP) info, and enable its AP."
    )
    parser.add_argument(
        "identifier",
        type=str,
        help="Last 4 digits of GoPro serial number, which is the last 4 digits of the default \
            camera SSID. (ex: GoPro 6665)",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the folder that contains ssid.txt (ex:  ./GoPro/GoPro XXXX/)",
        default=None,
    )
    args = parser.parse_args()
    path = Path(args.path) if args.path else None
    
    wifi_com = WifiCommunication(args.identifier, path)
    
    asyncio.run(main(wifi_com))