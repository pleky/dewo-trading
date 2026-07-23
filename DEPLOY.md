# Deploy to Fly.io

## Prereqs
- GitHub account
- Fly.io account (signup: https://fly.io/app/sign-up)
- `flyctl` installed
- `gh` CLI (optional, for repo create)

## 1. Install flyctl

```bash
curl -L https://fly.io/install.sh | sh
export FLYCTL_INSTALL="$HOME/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

## 2. Push to GitHub

```bash
cd /home/pleky/.openclaw/workspace/trading-journal
git init
git add .
git status  # AUDIT: pastikan CSV data TIDAK ke-stage
git commit -m "initial: streamlit trading dashboard"

# Option A: gh CLI
gh repo create dewo-trading --public --source=. --remote=origin --push

# Option B: manual
# Create repo di github.com dulu, lalu:
# git remote add origin git@github.com:USERNAME/dewo-trading.git
# git branch -M main
# git push -u origin main
```

## 3. Fly login + launch

```bash
fly auth login  # atau fly auth signup

# App sudah punya fly.toml, langsung deploy tanpa launch wizard:
fly apps create dewo-trading --org personal
fly volumes create trading_data --region sin --size 1 --yes
```

## 4. Deploy

```bash
fly deploy
```

## 5. Upload data awal ke volume

```bash
fly ssh sftp shell
# > cd /data
# > put 01-trade-log.csv
# > put 02-position-tracker.csv
# > put 05-monthly-summary.csv
# > put pending_orders.csv
# > put us_positions.csv
# > put cash_state.txt
# > put history.db
# > quit
```

Verify:
```bash
fly ssh console -C "ls -la /data"
```

## 6. Password gate (optional)

```bash
fly secrets set APP_PASSWORD=your_strong_password
```

Add to top of `web_screener.py`:
```python
import streamlit as st
if st.session_state.get("auth") != True:
    pw = st.text_input("Password", type="password")
    if pw and pw == st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD")):
        st.session_state.auth = True
        st.rerun()
    else:
        st.stop()
```

## 7. Open

```bash
fly open
# atau URL: https://dewo-trading.fly.dev
```

## Ops

```bash
fly logs           # stream logs
fly status         # health
fly ssh console    # shell into VM
fly deploy         # redeploy after code change
fly scale memory 1024  # bump RAM kalau OOM
```

## Backup data (recommended weekly)

```bash
fly ssh sftp shell
# > get /data/history.db ./backup/history-$(date +%F).db
# > get /data/02-position-tracker.csv ./backup/positions-$(date +%F).csv
```

Atau otomatis via cron lokal.
