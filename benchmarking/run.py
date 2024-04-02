import json
import time

import weaviate
import weaviate.classes as wvc

def run() -> None:
    client = weaviate.connect_to_local(
        skip_init_checks=True,
    )
    collection_name = "BenchmarkingT2Vrs"
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

    import_objects = 10000
    times = []
    with open("data/sphere.1M.jsonl", "r") as jsonl_file:
        for i, jsonl in enumerate(jsonl_file):
            json_parsed = json.loads(jsonl)
            start = time.time()
            collection.data.insert(
                properties={
                    "url": json_parsed["url"],
                    "title": json_parsed["title"],
                    "raw": json_parsed["raw"],
                    "sha": json_parsed["sha"],
                },
                uuid=json_parsed["id"],
                vector=json_parsed["vector"],
            )
            times.append(time.time() - start)
            if i % 1000 == 0:
                print(f"Imported {i} objects")
            if i == import_objects:
                break

    print(f"Imported {import_objects} objects in {sum(times)} seconds at an average of {sum(times) / len(times)} seconds per object")

    client.close()

if __name__ == "__main__":
    run()