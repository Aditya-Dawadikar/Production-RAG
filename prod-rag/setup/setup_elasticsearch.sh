#!/bin/bash
# Stage 2/3: Install Elasticsearch 8.x and populate the wiki_chunks index.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/_lib.sh"

log_stage "STAGE 2/3: Elasticsearch install & data population"

if systemctl is-active --quiet elasticsearch; then
    log_info "Elasticsearch service already running, skipping install"
else
    log_info "Adding Elastic 8.x apt repository..."

    wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | \
        sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg

    echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | \
        sudo tee /etc/apt/sources.list.d/elastic-8.x.list > /dev/null

    sudo apt-get update

    log_info "Installing Elasticsearch..."
    sudo apt-get install -y elasticsearch

    log_info "Disabling security and configuring single-node cluster..."

    sudo /usr/share/elasticsearch/bin/elasticsearch-keystore remove xpack.security.http.ssl.keystore.secure_password || true
    sudo /usr/share/elasticsearch/bin/elasticsearch-keystore remove xpack.security.transport.ssl.keystore.secure_password || true
    sudo /usr/share/elasticsearch/bin/elasticsearch-keystore remove xpack.security.transport.ssl.truststore.secure_password || true

    sudo mkdir -p /etc/elasticsearch/jvm.options.d

    cat <<EOF | sudo tee /etc/elasticsearch/jvm.options.d/heap.options > /dev/null
-Xms1g
-Xmx1g
EOF

    cat <<EOF | sudo tee /etc/elasticsearch/elasticsearch.yml > /dev/null
cluster.name: rag-es
node.name: rag-es-node-1

path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

network.host: 127.0.0.1
http.port: 9200

discovery.type: single-node

xpack.security.enabled: false
xpack.security.enrollment.enabled: false
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable elasticsearch
    sudo systemctl restart elasticsearch

    log_success "Elasticsearch installed and configured"
fi

log_info "Waiting for Elasticsearch to become available..."

for i in {1..30}; do
    if curl -s http://localhost:9200 > /dev/null; then
        log_success "Elasticsearch is up"

        curl -s -X PUT "http://localhost:9200/_all/_settings" \
            -H "Content-Type: application/json" \
            -d '{"index": {"number_of_replicas": 0}}' > /dev/null || true

        break
    fi

    if [ "$i" -eq 30 ]; then
        log_error "Elasticsearch did not become available in time"
        sudo systemctl status elasticsearch --no-pager -l
        sudo tail -n 100 /var/log/elasticsearch/rag-es.log
        exit 1
    fi

    sleep 5
done

source "$PROJECT_ROOT/venv/bin/activate"

log_info "Creating Elasticsearch index (if needed)..."
python "$SCRIPT_DIR/create_es_index.py"

log_info "Ingesting wiki chunks from S3 (if needed)..."
python "$SCRIPT_DIR/ingest_es_from_s3.py"

log_success "Elasticsearch ready"
