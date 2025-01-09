import time
import tempfile
import subprocess
import psutil
import os

def execute_code_with_timeout(code_string, timeout):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code_string)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        process = subprocess.Popen(['python', temp_file_name], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            raise TimeoutError("Code execution timed out")
        
        if process.returncode != 0:
            raise Exception(f"Code execution failed: {stderr.decode()}")
        
        return stdout.decode()
    finally:
        os.unlink(temp_file_name)