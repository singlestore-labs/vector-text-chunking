#!/usr/bin/env python3
# Copyright 2026 SingleStore, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
