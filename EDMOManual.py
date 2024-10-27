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
        
    
    def manualInput(self):
        # instructions to use: 
        # c motor_id 'amp'/'off'/'freq'/'phb' float (ex: c 3 amp 13.5)
        # f path (ex: cleanData/2024.09.24/Kumoko/12.02.51)
        
        instructions = input("Enter instructions:")
        command, detail = instructions.split(" ", 1)
        
        match (command):
            case "c":
                return "message", detail
            case "f":
                data = []
                for filename in os.listdir(detail):
                    pattern = r"^Motor[0-9]*\.log$"
                    if re.match(pattern, filename):
                        data.append(open(os.path.join(detail, filename), "r").read())

                return "path", data
            case _:
                print("wrong instruction flag")
                pass        
                
        return None, None
        
        
    async def run(self) -> None:
        await self.fusedCommunication.initialize()

        closed = False
        instruction = None
        
        try:
            while not closed:
                if self.activeSessions:
                    flag, instruction = self.manualInput()
                await self.update(instruction)
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
