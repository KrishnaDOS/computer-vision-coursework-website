"""Gunicorn configuration file"""

import multiprocessing

# Bind to all interfaces on port 8000
bind = "0.0.0.0:8000"

# Use gthread worker class for handling multiple threads per worker
worker_class = "gthread"

# Number of threads per worker
threads = 4

# timeout for worker restart
graceful_timeout = 30

# Keep alive connections (seconds)
keepalive = 2

# Timeout for a worker (seconds)
timeout = 600

# Recommended worker count: 2 * CPU cores + 1 (good general default)
workers = max(2, multiprocessing.cpu_count() * 2 + 1)

# Log settings
accesslog = '-'  # stdout
errorlog = '-'   # stderr
loglevel = 'info'

preload_app = False
