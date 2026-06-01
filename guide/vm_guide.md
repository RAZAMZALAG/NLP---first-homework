# Azure VM — `nlpgpu2025s-0004`

## Facts

| | |
|---|---|
| IP | `13.83.218.185` |
| User | `student` |
| Password | `Technion2025!` |
| OS | Ubuntu 22.04 DSVM |
| GPU | Tesla M60 (unused — LBFGS is CPU) |
| Project | `~/project/code_and_example` |

> Cost ~$1/hr Running. $0 when **Stopped (deallocated)**. Stop when done.

---

## Connect

```powershell
ssh student@13.83.218.185
```
Password: `Technion2025!`.

If timeout → Portal → VM → **Start**, wait 60s.
If host key conflict → `ssh-keygen -R 13.83.218.185`.

### Optional — passwordless

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\nlpvm" -N """"
type "$env:USERPROFILE\.ssh\nlpvm.pub" | ssh student@13.83.218.185 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

@"
Host nlpvm
    HostName 13.83.218.185
    User student
    IdentityFile ~/.ssh/nlpvm
"@ | Set-Content "$env:USERPROFILE\.ssh\config" -Encoding ASCII
```

Then: `ssh nlpvm`.

---

## Run Project

```bash
cd ~/project/code_and_example
uv run python code/main.py --sid 000000000 --model_number 1 --threshold 10
uv run python code/main.py --sid 000000000 --model_number 2 --threshold 10
```

`main.py` flags: `--sid`, `--model_number {1,2}`, `--threshold N`, `--lam X`.

> PDF says `--model`. Code uses `--model_number`.

Re-tag without retraining:
```bash
uv run python code/generate_comp_tagged.py --sid 000000000 --model_number 1
```

---

## Build + Check Submission

```bash
echo "%PDF-1.0" > report_000000000.pdf
zip HW1_000000000.zip -r code/ trained_models/ \
    comp_m1_000000000.wtag comp_m2_000000000.wtag report_000000000.pdf
uv run python submission_check.py --sid 000000000
```

Pass = `All checks passed`.

---

## Transfer Files

```powershell
# push
scp file student@13.83.218.185:~/project/code_and_example/code/
scp -r local_dir student@13.83.218.185:~/

# pull
scp student@13.83.218.185:~/project/code_and_example/comp_m1_*.wtag .
```

With alias: replace `student@13.83.218.185` with `nlpvm`.

---

## Stop VM

Portal → `nlpgpu2025s-0004` → **Stop** → confirm **Stopped (deallocated)**.

`sudo shutdown` from inside VM does NOT deallocate compute. Use Portal.

Auto-shutdown safety net: Portal → VM → **Operations → Auto-shutdown** → On → set time.

---

## First-Time Env Setup

Only if `.venv` or `pyproject.toml` missing:

```bash
cd ~/project/code_and_example
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cat > pyproject.toml <<'EOF'
[project]
name = "nlp-hw1"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = ["numpy","scipy","pandas","matplotlib","tqdm"]
EOF

uv sync
[ ! -e data ] && ln -s ~/project/data data
```

---

## Issues

| Symptom | Fix |
|---|---|
| Connection timeout | Portal Start. Or Networking → add inbound rule 22 from My IP. |
| Permission denied | Retype password (case-sensitive). |
| Host key changed | `ssh-keygen -R 13.83.218.185` |
| `uv: command not found` | `source $HOME/.local/bin/env` |
| `No pyproject.toml` | See First-Time Env Setup |
| Disk full | `df -h ~`, clean `trained_models/`, `uv cache prune` |
| Passphrase prompt | Regen key with 4-quote `-N """"` |

---

## End-of-Day

```bash
exit
```
Portal → **Stop** → confirm **Stopped (deallocated)**.
