# Azure VM Guide — `nlpgpu2025s-0004`

How to start, connect, use, and shut down the course Azure VM. Save the credit. Don't leave it running.

---

## 1. VM Facts

| Property | Value |
|---|---|
| Name | `nlpgpu2025s-0004` |
| OS | Ubuntu 22.04 Data Science VM (DSVM) |
| Size | Standard NV12s v3 — 12 vCPUs, 112 GiB RAM |
| GPU | NVIDIA Tesla M60, 8 GiB VRAM |
| Public IP | `13.83.218.185` |
| DNS | `nlpgpu2025s-0004.westus.cloudapp.azure.com` |
| Region | West US |
| SSH user | `student` |
| SSH password | `Technion2025!` |

> ⚠️ NV12s v3 ≈ $1+/hr while **Running**. Cost = 0 only when **Stopped (deallocated)**. Always stop when done.

---

## 2. First-Time Setup (Already Done — reference only)

1. Azure Portal → search `Virtual machines` → click `nlpgpu2025s-0004`.
2. **Start** button → wait until status = **Running**.
3. If SSH blocked (timeout): VM → **Networking** → **Add inbound port rule** → Source = `My IP`, Service = `SSH`, Port = `22`, Action = `Allow` → Save.
4. On Windows laptop PowerShell:
   ```powershell
   ssh student@13.83.218.185
   # yes to fingerprint
   # password: Technion2025!
   ```
5. On VM, install uv + create venv + symlink data (one-time):
   ```bash
   cd ~/project/code_and_example
   source $HOME/.local/bin/env
   uv sync          # installs deps into .venv
   [ ! -e data ] && ln -s ~/project/data data
   ```

---

## 3. Daily Workflow

### 3.1 Start VM (Portal)

Azure Portal → `nlpgpu2025s-0004` → **Start** → wait ~30–60 sec.

Public IP usually stays the same (`13.83.218.185`). If it changes, look at **Overview** page for new value.

### 3.2 SSH In

Windows PowerShell:
```powershell
ssh student@13.83.218.185
```
Password: `Technion2025!`

### 3.3 Optional — passwordless SSH (do once, save time forever)

On laptop:
```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\nlpvm -N '""'
type $env:USERPROFILE\.ssh\nlpvm.pub | ssh student@13.83.218.185 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```
Add to `$env:USERPROFILE\.ssh\config` (create if missing):
```
Host nlpvm
    HostName 13.83.218.185
    User student
    IdentityFile ~/.ssh/nlpvm
```
Then just:
```powershell
ssh nlpvm
```
No password prompt.

### 3.4 Activate Project Env Inside VM

Every fresh SSH session:
```bash
cd ~/project/code_and_example
# uv handles venv automatically via `uv run`
```

To run any command:
```bash
uv run python code/main.py --sid <id> --model_number 1 --threshold 10
```

Or activate manually:
```bash
source .venv/bin/activate
python code/main.py ...
deactivate     # exit venv
```

---

## 4. File Transfer

### 4.1 Push files from laptop → VM

```powershell
# single file
scp "c:\Users\razam\Documents\Degree\NLP\project\code_and_example\code\preprocessing.py" student@13.83.218.185:~/project/code_and_example/code/

# whole project
scp -r "c:\Users\razam\Documents\Degree\NLP\project" student@13.83.218.185:~/
```

### 4.2 Pull files from VM → laptop

```powershell
# pull tagged output
scp student@13.83.218.185:~/project/code_and_example/comp_m1_000000000.wtag "c:\Users\razam\Documents\Degree\NLP\project\"

# pull whole code dir
scp -r student@13.83.218.185:~/project/code_and_example/code "c:\Users\razam\Documents\Degree\NLP\project\"
```

### 4.3 With passwordless SSH config (`nlpvm` host)

```powershell
scp file nlpvm:~/project/
scp -r nlpvm:~/project/code_and_example/trained_models .
```

---

## 5. Verify GPU / Env

Inside VM:
```bash
nvidia-smi          # GPU status
python3 --version   # python from venv
uv pip list         # installed packages
df -h ~             # disk usage
free -h             # RAM usage
nproc               # CPU cores
```

---

## 6. Stop VM — CRITICAL

### 6.1 Portal way (correct way)

Azure Portal → `nlpgpu2025s-0004` → **Stop** button → confirm.

Status must reach **Stopped (deallocated)**. Cost during this state = $0 (disk only, negligible).

### 6.2 From inside VM (incomplete — still pay for compute!)

```bash
sudo shutdown -h now      # halts OS but Azure may not deallocate
```
This brings VM to **Stopped** but possibly not **deallocated**. Always confirm via Portal that status = **Stopped (deallocated)**. If only **Stopped**, click **Stop** in Portal.

### 6.3 Auto-shutdown (set once, save credit)

Portal → VM → left menu → **Operations** → **Auto-shutdown** → toggle **On** → set time (e.g. `02:00`) + timezone → leave email blank or add yours → **Save**.

Now VM auto-stops every day at chosen time. Still need to manually start it.

---

## 7. Common Issues

| Problem | Fix |
|---|---|
| `ssh: connect to host 13.83.218.185 port 22: Connection timed out` | VM not started, or NSG blocks port 22. Portal → Start. Or Networking → add inbound SSH rule for your IP. |
| `Permission denied (publickey,password)` | Wrong password, or SSH key mismatch. Re-type `Technion2025!` carefully (case-sensitive). |
| Public IP changed | Portal Overview → copy new IP. Update `~/.ssh/config` Host entry. |
| `nvidia-smi: command not found` | Driver missing. Course DSVM has it. If broken: `sudo apt install nvidia-driver-535`. |
| `uv: command not found` | Run `source $HOME/.local/bin/env` then retry. Or run `bash init.sh` again. |
| Disk full | `df -h ~`; clean old pickles in `trained_models/`, or `uv cache prune`. |
| SCP hangs / very slow | Network throttling. Retry. Or use `rsync -avz --progress`. |
| Forgot password | Portal → VM → **Reset password** (left menu under Help). |

---

## 8. Quick Reference Cheat Sheet

```powershell
# Laptop side
ssh student@13.83.218.185              # connect
scp file student@13.83.218.185:~/      # push file
scp student@13.83.218.185:~/file .     # pull file
```

```bash
# VM side
cd ~/project/code_and_example
source $HOME/.local/bin/env             # PATH for uv (only if missing)
uv run python code/main.py --sid 000000000 --model_number 1 --threshold 10
nvidia-smi
exit                                    # close SSH
```

```
# Portal — every session end
VM page → Stop button → confirm Stopped (deallocated)
```

---

## 9. End-of-Day Checklist

1. Save / push code (`scp` or `git push`).
2. `exit` from SSH.
3. Azure Portal → **Stop** VM.
4. Verify status = **Stopped (deallocated)**.

Skip step 3 = burn credit overnight. Auto-shutdown is your safety net, not primary control.
