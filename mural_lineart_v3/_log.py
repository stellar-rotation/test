import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=39249, username="root",
          password="Tb6cW5iVP6lo", look_for_keys=False, allow_agent=False, timeout=15)

# Full Val L1
stdin, stdout, stderr = c.exec_command("grep 'Val L1' /root/mural_lineart_v3/train.log")
print("=== Full Val L1 history ===")
for line in stdout.read().decode().strip().split("\n"):
    print(line.strip())

# First NaN traces
stdin, stdout, stderr = c.exec_command("grep -n 'nan' /root/mural_lineart_v3/train.log | head -5")
print("\n=== First NaN occurrences ===")
for line in stdout.read().decode().strip().split("\n"):
    # Pick out key info from tqdm lines
    if "D=" in line and "G=" in line:
        idx = line.find("D=")
        print("  " + line[:10] + "..." + line[idx:idx+40])
    else:
        print(line.strip()[:200])

c.close()
