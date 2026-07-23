# Trading Journal — Dewo

## Struktur File

| File | Isi |
|------|-----|
| `01-trade-log.csv` | Log semua transaksi (buy/sell) — 1 baris per trade |
| `02-position-tracker.csv` | Snapshot posisi saham saat ini (update mingguan) |
| `03-rules-checklist.md` | Aturan trading — baca ulang tiap beli saham |
| `04-reflection-template.md` | Refleksi per trade — isi setelah exit |
| `05-monthly-summary.csv` | Rekap bulanan performa |

## Cara Import ke Google Sheets

1. Buka [sheets.google.com](https://sheets.google.com)
2. New spreadsheet
3. File → Import → Upload → pilih CSV
4. Setting: "Replace current sheet" atau "New sheet"
5. Delimiter: Comma
6. Import data

## Cara Import ke Excel

1. Buka Excel
2. Data → From Text/CSV → pilih file
3. Delimiter: Comma
4. Load

## Alur Kerja Harian

**Sebelum trading (pagi):**
1. Buka `03-rules-checklist.md`, baca ulang
2. Cek `02-position-tracker.csv` — apa yang perlu action?
3. Kalau mau beli saham baru → jawab Anti-FOMO checklist

**Setelah trade:**
1. Update `01-trade-log.csv` — tambah baris baru
2. Update `02-position-tracker.csv` — refresh angka
3. Isi `04-reflection-template.md` — refleksi trade

**Setiap Minggu (Minggu malam):**
1. Update `02-position-tracker.csv` dengan harga close Jumat
2. Baca refleksi trade minggu ini
3. Cari pola kesalahan

**Setiap Akhir Bulan:**
1. Update `05-monthly-summary.csv`
2. Hitung win rate, P/L, R:R
3. Set target bulan depan
4. Rebalance kalau perlu

## Metrik Penting

- **Win Rate** = winning trades / total trades × 100%
  - Target: > 50% untuk swing, > 40% untuk scalp
  - Bisa < 50% asal Risk Reward Ratio bagus

- **Risk Reward Ratio** = avg win / avg loss
  - Target: minimum 2:1 (untung 2x lebih besar dari rugi)
  - Ideal: 3:1

- **Max Drawdown** = penurunan terbesar dari peak portfolio
  - Kalau > 25%, STOP trading, evaluasi ulang

- **Return %** = (nilai akhir - nilai awal) / nilai awal × 100%
  - Benchmark IHSG: ~8-12%/tahun
  - Target realistis pribadi: 15-20%/tahun (dengan risk moderat)

## Recovery Mode Progress

**Starting point (22 Jul 2026):**
- Portfolio value: Rp 49.580.501
- Total invested: Rp 73.666.205
- Floating loss: -Rp 24.136.397 (-32.74%)
- Realized loss (IKAN 50% sell): -Rp 2.953.990

**Target 12 bulan:**
- Break-even portfolio: Rp 73.666.205
- Perlu return: +48% dari nilai sekarang
- Realistis: 12-18 bulan dengan blue chip DCA

**Target 6 bulan:**
- Portfolio value: Rp 55-58 juta (recovery +10-15%)
- Setidaknya berhenti pendarahan
- Blue chip position established
