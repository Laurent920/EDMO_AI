import asyncio
from asyncio import tasks
from typing import AsyncIterator
from aiohttp import web
from aiohttp.web_middlewares import normalize_path_middleware
from aiohttp_middlewares import cors_middleware
from prompt_toolkit import Application  # type: ignore
from EDMOSession import EDMOSession
from FusedCommunication import FusedCommunication, FusedCommunicationProtocol
from aiortc.contrib.signaling import object_from_string, object_to_string
import os
import re
import aioconsole



class EDMOManual:
    def __init__(self):
        self.activeEDMOs: dict[str, FusedCommunicationProtocol] = {}
        self.activeSessions: dict[str, EDMOSession] = {}

        self.fusedCommunication = FusedCommunication()
        self.fusedCommunication.onEdmoConnected.append(self.onEDMOConnected)
        self.fusedCommunication.onEdmoDisconnected.append(self.onEDMODisconnect)

        self.simpleViewEnabled = False
    
    # region EDMO MANAGEMENT

    def onEDMOConnected(self, protocol: FusedCommunicationProtocol):
        # Assumption: protocol is non null
        identifier = protocol.identifier
        self.activeEDMOs[identifier] = protocol
        
        print("Edmo " + identifier + " connected") 
        self.activeSessions[identifier] = EDMOSession(
            protocol, 4, self.removeSession
        )

    def onEDMODisconnect(self, protocol: FusedCommunicationProtocol):
        # Assumption: protocol is non null
        identifier = protocol.identifier

        # Remove session from candidates
        print("Edmo " + identifier + " disconnected") 
        if identifier in self.activeEDMOs:
            del self.activeEDMOs[identifier]
            
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
            sessionUpdates.append(asyncio.create_task(session.update(instruction)))

        # Ensure that the update cycle runs at most 10 times a second
        minUpdateDuration = asyncio.create_task(asyncio.sleep(0.1))

        await serialUpdateTask
        if len(sessionUpdates) > 0:
            await asyncio.wait(sessionUpdates)
        await minUpdateDuration
    
    
    def reset(self):
        for sessionID in self.activeSessions:
            session = self.activeSessions[sessionID] 
            for motor in session.motors:
                motor._amp = 0.0
                motor._freq = 0.0
                motor._offset = 90
                motor._phaseShift = 0
    
    
    async def manualInput(self):
        # instructions to use: 
        # c motor_id 'amp'/'off'/'freq'/'phb' float (ex: c 3 amp 13.5)
        # f path (ex: cleanData/2024.09.24/Kumoko/12.02.51)
        while(True):
            instructions = await asyncio.gather(
                            aioconsole.ainput(),
                        )
            instruction = instructions[0].split(" ", 1)
            
            data = None
            match (instruction[0]):
                case "c":
                    print(f'Sending instruction: {instruction[1]}')
                    motorNumber, command = instruction[1].split(" ", 1)
                    for sessionID in self.activeSessions:
                        session = self.activeSessions[sessionID] 
                        session.updateMotor(int(motorNumber), command)
                case "f":
                    data = {}
                    nbInstructions = 0
                    for filename in os.listdir(instruction[1]):
                        pattern = r"^Motor[0-9]*\.log$"
                        if re.match(pattern, filename):
                            data[filename[5]] = (open(os.path.join(instruction[1], filename), "r").read()).splitlines()
                            nbInstructions = len(data[filename[5]])
                            
                    print("reading file...")    
                    print(nbInstructions)
                    for i in range(1, nbInstructions):
                        for id, values in data.items():
                            await self.update(self.dataToInstructions(id, values[i]))
                    print("finished reading file")
                case "reset":
                    print("Resetting")
                    self.reset()
                case _:
                    print("wrong instruction flag")
                    pass        
            
    
    def dataToInstructions(self, id, data):
        splits = data.split(',')
        instruction = [f'{id} freq{splits[1]}']
        instruction.append(f'{id} amp{splits[2]}')
        instruction.append(f'{id} off{splits[3]}')
        instruction.append(f'{id} phb{splits[4]}')
        return instruction
        
    async def run(self) -> None:
        await self.fusedCommunication.initialize()

        closed = False
        asyncio.get_event_loop().create_task(self.manualInput())

        try:
            while not closed:   
                await self.update()
        except (asyncio.exceptions.CancelledError, KeyboardInterrupt):
            pass

    async def onShutdown(self, app: web.Application | None = None):
        yield

        print("Cleaning up")
        """Shuts down existing connections gracefully to prevent a minor deadlock when shutting down the server"""
        self.fusedCommunication.close()
        for s in [sess for sess in self.activeSessions]:
            session = self.activeSessions[s]
            await session.close()
        pass
    
async def main():
    server = EDMOManual()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
