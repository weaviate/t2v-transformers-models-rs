import concurrent.futures
import json
import time
import typing

import weaviate
import weaviate.classes as wvc

def insert(collection: weaviate.collections.Collection, obj: dict) -> float:
    start = time.time()
    collection.data.insert(
        properties={
            "url": obj["url"],
            "title": obj["title"],
            "raw": obj["raw"],
            "sha": obj["sha"],
        },
        # uuid=obj["id"],
        # vector=obj["vector"],
    )
    return time.time() - start

def run() -> None:
    client = weaviate.connect_to_local(
        skip_init_checks=True,
    )
    collection_name = "BenchmarkingT2Vrs"
    try:
        client.collections.delete(collection_name)

        collection = client.collections.create(
            collection_name,
            properties=[
                wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="raw", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="sha", data_type=wvc.config.DataType.TEXT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(vectorize_collection_name=False),
        )

        import_objects = 100000
        futures: typing.List[concurrent.futures.Future] = []

        with open("data/sphere.1M.jsonl", "r") as jsonl_file:
            with collection.batch.dynamic() as batch:
                for i, jsonl in enumerate(jsonl_file):
                    json_parsed = json.loads(jsonl)
                    batch.add_object(properties={
                        "url": json_parsed["url"],
                        "title": json_parsed["title"],
                        "raw": json_parsed["raw"],
                        "sha": json_parsed["sha"],
                    })
                    if (i + 1) % 10000 == 0:
                        print(f"Scheduled {i} objects")
                        print("Flushing batch")
                        batch.flush()
                    if i == import_objects:
                        break
        
        times: typing.List[float] = []
        for future in concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION).done:
            took = future.result()
            print(f"Imported object in {took} seconds")
            times.append(took)
        print(f"Imported {import_objects} objects in {sum(times)} seconds at an average of {sum(times) / len(times)} seconds per object")
    finally:
        client.close()

if __name__ == "__main__":
    run()