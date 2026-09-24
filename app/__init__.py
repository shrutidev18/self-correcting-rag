import sys
import platform

# pysqlite3 patch only needed inside Docker (Linux)
# On Windows/Mac, system SQLite is already new enough for ChromaDB
if platform.system() == "Linux":
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass