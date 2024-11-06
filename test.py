import asyncio


async def sleeper(i):
    await asyncio.sleep(i)

async def test():
    while True:
        await sleeper(2)

        
def main():
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(test())
    print("task done")
    
    

if __name__ == "__main__":
    main()