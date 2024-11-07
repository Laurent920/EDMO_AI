import asyncio
from EDMOSession import EDMOSession
from FusedCommunication import FusedCommunication, FusedCommunicationProtocol
import os
import re
import time
import aioconsole
from Utilities.Helpers import toTime
from datetime import datetime, timedelta
from pathlib import Path
from GoPro.COHN.communicate_via_cohn import COHNCommunication
from GoPro.wifi.WifiCommunication import WifiCommunication


class EDMOManual:
    def __init__(self):
        self.activeEDMOs: dict[str, FusedCommunicationProtocol] = {}
        self.activeSessions: dict[str, EDMOSession] = {}
        self.goPros: dict[str, WifiCommunication] = {}

        self.fusedCommunication = FusedCommunication()
        self.fusedCommunication.onEdmoConnected.append(self.onEDMOConnected)
        self.fusedCommunication.onEdmoDisconnected.append(self.onEDMODisconnect)
        
        self.protocol:FusedCommunicationProtocol
        self.dataPath:str = None

        self.simpleViewEnabled = False
        
        self.connected = asyncio.Condition()
        self.initialized = False
        self.closed = False
         # GoPro 
        folderPath = "./GoPro/"
        for folderName in os.listdir(folderPath):
            pattern = r"GoPro 6665" #\d{4}"
            if re.match(pattern, folderName):
                print(f"Getting credentials from : {folderPath}{folderName}/\t in EDMOManual init")
                self.goPros[folderName] = WifiCommunication(folderName,
                    Path(f"{folderPath}{folderName}/"))
                
        asyncio.get_event_loop().create_task(self.fusedCommunication.initialize())

    
    async def initialize(self, dataPath:str = None):
        self.closed = False
        self.initialized = True
        
        print(f"Initializing with data path: {dataPath}")
        self.dataPath = dataPath
            
        self.onEDMOConnected(self.protocol)
    # region EDMO MANAGEMENT

    def onEDMOConnected(self, protocol: FusedCommunicationProtocol):
        # Assumption: protocol is non null
        self.protocol = protocol
        identifier = protocol.identifier
        self.activeEDMOs[identifier] = protocol
        
        print("Edmo " + identifier + " connected") 
        self.activeSessions[identifier] = EDMOSession(
            protocol, 4, self.removeSession, self.dataPath
        )
        
        # TODO test how much delay we get between session and gopro
        for gopro_id in self.goPros:
            print(f"Sending start command to gopro in edmoConnected {gopro_id}")
            self.goPros[gopro_id].send_command('start')
        
        for i in range(4):
            self.activeSessions[identifier].registerManualPlayer()
        
        asyncio.create_task(self._notify())

    async def _notify(self):
        async with self.connected:
            self.connected.notify_all()

    def onEDMODisconnect(self, protocol: FusedCommunicationProtocol):
        # Assumption: protocol is non null
        identifier = protocol.identifier

        # Remove session from candidates
        print("Edmo " + identifier + " disconnected") 
        if identifier in self.activeEDMOs:
            del self.activeEDMOs[identifier]
            
    def GoProOff(self):
        for gopro_id in self.goPros:
            self.goPros[gopro_id].send_command('stop')
            
    def logVideoFile(self):
        for gopro_id in self.goPros:
            self.goPros[gopro_id].send_command('get last video', self.dataPath)
            
    def removeSession(self, session: EDMOSession):
        identifier = session.protocol.identifier
        if identifier in self.activeSessions:
            del self.activeSessions[identifier]
            
    async def update(self, instruction=None):
        """Standard update loop to be performed at most 10 times a second"""
        # Update the serial stuff
        serialUpdateTask = asyncio.create_task(self.fusedCommunication.update())

        # Update all sessions
        sessionUpdates = []

        for sessionID in self.activeSessions:
            session = self.activeSessions[sessionID]
            sessionUpdates.append(asyncio.create_task(session.update()))

        # Ensure that the update cycle runs at most 10 times a second
        minUpdateDuration = asyncio.create_task(asyncio.sleep(0.1))

        await serialUpdateTask
        if len(sessionUpdates) > 0:
            await asyncio.wait(sessionUpdates)
        await minUpdateDuration
    
    
    def reset(self):
        for sessionID in self.activeSessions:
            session = self.activeSessions[sessionID] 
            for player in session.activePlayers:
                player.reset()
                
                
    async def runInputFile(self, id, data:list[str], nbInstructions):
        print(f"reading player {id} ...")  
        for i in range(nbInstructions - 1):
            cur_split = data[i].split(': ')
            next_split = data[i+1].split(': ')
            
            c = datetime.strptime(cur_split[0],"%H:%M:%S.%f")
            n = datetime.strptime(next_split[0],"%H:%M:%S.%f")
            cur_timedelta = timedelta(hours=c.hour, minutes=c.minute, seconds=c.second)
            if i == 0:
                delay = cur_timedelta.total_seconds()
                print(delay)
                await asyncio.sleep(delay)
            else:
                delay = (n-c).total_seconds()
                if delay > 10.0:
                    print(delay)
                await asyncio.sleep(delay)
            
            for sessionID in self.activeSessions:
                session = self.activeSessions[sessionID]
                await session.activePlayers[int(id)].onMessage(cur_split[1])
        await asyncio.sleep(5)
        print(f"finished reading player {id}")
    
    
    async def parseInputFile(self, instructions):
        async with self.connected:
            print("waiting for active sessions 165 EDMOManual.py")
            await self.connected.wait_for(lambda: bool(self.activeSessions))
        # instructions must be of type: 
        # c motor_id 'amp'/'off'/'freq'/'phb' float (ex: c 3 amp 13.5)
        # f path (ex: cleanData/2024.09.24/Kumoko/12.02.51)
        instruction = instructions.split(" ", 1)
        data = None
        match (instruction[0]):
            case "c":
                print(f'Sending instruction: {instruction[1]}')
                motorNumber, command = instruction[1].split(" ", 1)
                for sessionID in self.activeSessions:
                    session = self.activeSessions[sessionID] 
                    await session.activePlayers[int(motorNumber)].onMessage(command)
            case "f":
                filepath = instruction[1]
                data = {}
                nbInstructions = {}
                filepath = os.path.abspath(filepath)
                for filename in os.listdir(filepath):
                    pattern = r"^Input_Player[0-9]*\.log$"
                    if re.match(pattern, filename):
                        data[filename[12]] = (open(os.path.join(filepath, filename), "r").read()).splitlines()
                        nbInstructions[filename[12]] = (len(data[filename[12]]) - 1)
                if not data:    
                    print("No Input_Player in this folder ==> ending the run")
                    
                loop = asyncio.get_event_loop()
                tasks = []
                for key in data.keys():
                    task = loop.create_task(self.runInputFile(key, data[key], nbInstructions[key]))
                    tasks.append(task)
                await asyncio.gather(*tasks)
                return True
            case "reset":
                print("Resetting")
                self.reset()
            case "help":
                print("""instructions to use: 
                    single control: c motor_id 'amp'/'off'/'freq'/'phb' float (ex: c 3 amp 13.5) 
                    replay file   : f path (ex: cleanData/2024.09.24/Kumoko/12.02.51)""")
            case _:
                print("wrong instruction flag")
                pass      
    
    
    async def manualInput(self):
        while len(self.activeEDMOs) < 1:
            await asyncio.sleep(1)
            
        while(True):
            instructions = await asyncio.gather(
                            aioconsole.ainput('Enter command or file to read (Enter help for help): '),
                        )
            instruction = instructions[0]
                
            await self.parseInputFile(instruction)      
        
        
    async def run(self) -> None:
        if self.dataPath:
            replayFile = asyncio.get_event_loop().create_task(self.parseInputFile("f " + self.dataPath))
        if self.initialized:
            await self._notify()
        
        try:
            while not self.closed: 
                await self.update()
                if self.dataPath and replayFile.done():
                    self.closed = True
        except (asyncio.exceptions.CancelledError, KeyboardInterrupt):
            await self.onShutdown()
            pass
        await self.close()
        
    async def close(self):
        self.closed = True
        for s in [sess for sess in self.activeSessions]:
            session = self.activeSessions[s]
            await session.close()
        
        self.GoProOff()
        time.sleep(2)
        self.logVideoFile()
            

    async def onShutdown(self, app: None = None):
        print("Cleaning up")
        """Shuts down existing connections gracefully to prevent a minor deadlock when shutting down the server"""
        self.close()

        self.fusedCommunication.close()
        await asyncio.sleep(3)
        pass
    
async def main():
    server = EDMOManual()
    server.initialize()
    asyncio.get_event_loop().create_task(server.manualInput())
    await server.run()


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
