import concurrent.futures
import json
import requests
import time
import typing


PORT = 3000

def send(obj: typing.Dict[str, typing.Any], s: requests.Session) -> float:
    start = time.time()
    s.post(f"http://localhost:{PORT}/vectors", json={
        "text": obj["raw"],
    })
    taken = time.time() - start
    print(f"Retrieved vector in {taken:.2f} seconds")
    return taken


def run(objs: typing.List[str]) -> None:
    session = requests.Session()
    for obj in objs:
        send(json.loads(obj), session)

if __name__ == "__main__":
    futures: typing.List[concurrent.futures.Future] = []
    objs: typing.List[str] = []
    with open("data/sphere.1M.jsonl", "r") as file:
        for i, obj in enumerate(file):
            objs.append(obj)
            if (i + 1) % 10000 == 0:
                break
    print(f"Loaded {len(objs)} objects")
    print("Starting benchmark")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(4):
            futures.append(executor.submit(run, objs))
        print("Scheduled all tasks, waiting for completion")
        concurrent.futures.wait(futures)