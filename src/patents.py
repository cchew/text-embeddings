'''
Trying out Patents embedding in Elasticsearch.

Modified from main.py
'''
import getopt
import json
import time
import sys

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Use tensorflow 1 behavior to match the Universal Sentence Encoder
# examples (https://tfhub.dev/google/universal-sentence-encoder/2).
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

# Start web server
from flask import Flask,request,jsonify
from flask_cors import CORS

##### SERVING #####

app = Flask(__name__)
CORS(app)

def serve():
    '''Make search available'''
    app.run(host='127.0.0.1', port=8085, debug=True)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('text')
    response, embedding_time, search_time = _query(query)
    
    return jsonify({
        'response': response,
        'embeddingTime': embedding_time,
        'searchTime': search_time
    })

##### INDEXING #####

def index_data(delete_existing=False):
    if delete_existing:
        print('Deleting existing {0} index'.format(INDEX_NAME))
        client.indices.delete(index=INDEX_NAME, ignore=[404])

        print("Creating the '{0}' index.".format(INDEX_NAME))
        with open(INDEX_FILE) as index_file:
            source = index_file.read().strip()
            client.indices.create(index=INDEX_NAME, body=source)

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")

def index_batch(docs):
    claims = [doc["claim"] for doc in docs]
    claim_vectors = embed_text(claims)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["claim_vector"] = claim_vectors[i]
        requests.append(request)
    bulk(client, requests)

##### SEARCHING #####

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return

def handle_query():
    query = input("Enter query: ")

    response, embedding_time, search_time = _query(query)

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()

def _query(query):
    '''Helper method for running query'''
    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['claim_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["applicationNumber", "title", "date", "claim"]}
        }
    )
    search_time = time.time() - search_start

    return response, embedding_time, search_time

##### EMBEDDING #####

def embed_text(text):
    vectors = session.run(embeddings, feed_dict={text_ph: text})
    return [vector.tolist() for vector in vectors]

##### MAIN SCRIPT #####

if __name__ == '__main__':
    INDEX_NAME = "patents"
    INDEX_FILE = "data/patents/index.json"
    DATA_FILE = "data/patents/patents.json"

    BATCH_SIZE = 1000
    SEARCH_SIZE = 21
    GPU_LIMIT = 0.5

    # Process argument options
    run_index, run_serve, delete_index = False, False, False
    try:
        opts, args = getopt.getopt(sys.argv[1:],"isf:d",["index","serve","file=","delete"])
    except getopt.GetoptError:
        print('patents.py [-i] [-s] [-f <file_name>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--index"):
            run_index = True
        elif opt in ("-s", "--serve"):
            run_serve = True
        elif opt in ("-d", "--delete"):
            delete_index = True
        elif opt in ("-f", "--file"):
            DATA_FILE = arg

    print("Running index: {0}, serve: {1}, delete existing index: {3}, data file: {2}".format(run_index, run_serve, DATA_FILE, delete_index))

    print("Downloading pre-trained embeddings from tensorflow hub...")
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    text_ph = tf.placeholder(tf.string)
    embeddings = embed(text_ph)
    print("...done downloading pre-trained embeddings!")

    print("Creating tensorflow session...")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_LIMIT
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print("...done creating tensorflow session!")

    client = Elasticsearch(timeout=60)

    if run_index:
        index_data(delete_index)
    #run_query_loop()
    if run_serve:
        serve()

    print("Closing tensorflow session...")
    session.close()
    print("...done closing tensorflow session!")
