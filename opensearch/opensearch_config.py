#!/usr/bin/env python3
"""OpenSearch connection configuration."""

from opensearchpy import OpenSearch

def get_client(password=None):
    """Get OpenSearch client with proper auth."""
    # For development without security
    if password is None:
        return OpenSearch(
            hosts=['http://localhost:9200']
        )
    # For production with security
    return OpenSearch(
        hosts=['https://localhost:9200'],
        http_auth=('admin', password),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False
    )

# Test connection
if __name__ == "__main__":
    client = get_client()
    info = client.info()
    print(f"✅ Connected to OpenSearch {info['version']['number']}")
    print(f"Cluster name: {info['cluster_name']}")