
from paramiko import SSHClient
import os
import time

#ssh 6835384@gemini.science.uu.nl

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def submit_command(command):
    print("Submitting command: {}".format(command))
    stdin, stdout, stderr = client.exec_command(command)
    time.sleep(1)
    stdout = stdout.readlines()
    if stdout != []:
        for line in stdout:
            print(line)
    stderr = stderr.readlines()
    if stderr != []:
        print("There was an error:")
        for line in stderr:
            print(line)

if __name__ == "__main__":
    client = SSHClient()
    client.load_system_host_keys()

    print("Connecting via SSH to cluster...")
    client.connect(hostname='gemini.science.uu.nl', username='6835384')
    print("Connection successful!")

    COMMAND = "SUBMIT" # Submit the jobs listed in submission_queue.sh
    #COMMAND = "QSTAT" # Check out the cluster queue

    if COMMAND == "SUBMIT":
        os.system("sh " + DIR_PATH + "/push.sh")
        time.sleep(5)
        submit_command('. start.sh && cd /nethome/6835384/dmrg/cluster/ && bash --login -c "sh submission_queue.sh"')
    elif COMMAND == "QSTAT":
        submit_command('bash --login -c "qstat -f"')
    
    print("Closing the connection...")
    client.close()
    print("Connection closed!")
