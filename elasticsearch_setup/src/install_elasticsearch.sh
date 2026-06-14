#!/bin/bash
set -e

sudo apt-get update
sudo apt-get install -y openjdk-21-jdk curl wget gnupg

# Elastic GPG Key
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | \
sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg

# Elastic Repo
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | \
sudo tee /etc/apt/sources.list.d/elastic-8.x.list

sudo apt-get update

# Install ES
sudo apt-get install -y elasticsearch

# Remove auto-generated security secrets if they exist
sudo /usr/share/elasticsearch/bin/elasticsearch-keystore remove xpack.security.http.ssl.keystore.secure_password || true
sudo /usr/share/elasticsearch/bin/elasticsearch-keystore remove xpack.security.transport.ssl.keystore.secure_password || true
sudo /usr/share/elasticsearch/bin/elasticsearch-keystore remove xpack.security.transport.ssl.truststore.secure_password || true

# Small instance heap
sudo mkdir -p /etc/elasticsearch/jvm.options.d

cat <<EOF | sudo tee /etc/elasticsearch/jvm.options.d/heap.options
-Xms1g
-Xmx1g
EOF

# Fresh config
cat <<EOF | sudo tee /etc/elasticsearch/elasticsearch.yml
cluster.name: rag-es
node.name: rag-es-node-1

path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

network.host: 0.0.0.0
http.port: 9200

discovery.type: single-node

xpack.security.enabled: false
xpack.security.enrollment.enabled: false
EOF

sudo systemctl daemon-reload
sudo systemctl enable elasticsearch
sudo systemctl restart elasticsearch

echo "Waiting for Elasticsearch..."

for i in {1..30}; do
    if curl -s http://localhost:9200 > /dev/null; then
        echo "Elasticsearch is ready"

        curl -X PUT "http://localhost:9200/_all/_settings" \
        -H "Content-Type: application/json" \
        -d '{
          "index": {
            "number_of_replicas": 0
          }
        }' || true

        curl http://localhost:9200
        exit 0
    fi

    sleep 5
done

echo "Failed to start Elasticsearch"
sudo systemctl status elasticsearch --no-pager -l
sudo tail -n 100 /var/log/elasticsearch/rag-es.log

exit 1