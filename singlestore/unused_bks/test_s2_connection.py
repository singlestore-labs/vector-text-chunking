#!/usr/bin/env python3
"""Test SingleStore cloud connection."""

import pymysql
import sys
import os

# Connection details
host = 'svc-aac188a1-2ed7-420b-9316-1fc3b9592536-dml.aws-oregon-4.svc.singlestore.com'
port = 3306
user = 'admin'
password = os.environ.get('SINGLESTORE_PASSWORD', '')

# Check if password is set
if not password:
    print("Error: SINGLESTORE_PASSWORD environment variable not set")
    print("Please set it with: export SINGLESTORE_PASSWORD='your_password'")
    sys.exit(1)

try:
    print("Connecting to SingleStore cloud...")
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        autocommit=True
    )

    print("✅ Connected successfully!")

    # List databases
    with connection.cursor() as cursor:
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        print("\nAvailable databases:")
        for db in databases:
            print(f"  - {db[0]}")

    connection.close()
    print("\nConnection test successful!")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)