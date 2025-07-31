# Shared in-memory backend logs for user display (no file I/O)
backend_logs = []
MAX_LOGS = 200
 
def add_backend_log(message):
    print(message)
    backend_logs.append(message)
    if len(backend_logs) > MAX_LOGS:
        backend_logs.pop(0) 