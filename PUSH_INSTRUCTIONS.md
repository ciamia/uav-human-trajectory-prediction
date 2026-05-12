# Push to GitHub — step by step

Bu dosya **bir defalık** kullanım için. GitHub'a yükledikten sonra silebilirsin.

---

## 0. Önceden

Bu repo `StanfordASL/Trajectron-plus-plus`'ın bir fork'u olarak başladı. Senin değişikliklerini Stanford'ın deposuna pushlayamazsın — kendi deponu açman lazım. İki seçenek:

- **A. Tamamen yeni bir kendi repon** (önerilen, tez için temiz).
- **B. Stanford'unkini fork** edip push.

A daha temiz çünkü tez başlığını, kendi commit history'ni vs. yönetebilirsin. Aşağısı A'yı varsayar.

---

## 1. GitHub'da yeni repo aç

1. https://github.com/new adresine git.
2. Doldur:
   - **Repository name:** `uav-human-trajectory-prediction` (veya istediğin)
   - **Description:** `Master thesis — human trajectory prediction for UAV safety, with VIO-based ground truth`
   - **Visibility:** `Private` (tez bitene kadar private tut; sonra public yaparsın)
   - `Initialize this repository with a README` → **KAPALI BIRAK** (zaten README'miz var)
   - `.gitignore` → **None** (zaten ekledik)
   - License → şimdilik **None**, sonra eklersin
3. `Create repository` bas.

Şimdi sayfada şuna benzeyen bir adres göreceksin:

```
https://github.com/<KULLANICI_ADIN>/uav-human-trajectory-prediction.git
```

Onu bir yere kopyala. Aşağıda `<URL>` yerine koyacağız.

---

## 2. Worktree → ana branch'e geç ve commit hazırla

Şu an `(no branch)` durumdasın. Önce normal bir branch'e geç.

```bash
cd /Users/simayyalcin/.cursor/worktrees/Trajectron-plus-plus/wrz

# durumu gör
git status

# README değişikliklerini ve docs/'u commit edeceğiz
# .gitignore zaten büyük modelleri/videoları dışlayacak

# yeni bir branch oluştur
git checkout -b thesis/main

# tüm değişiklikleri ekle
git add .gitignore README.md TRAJECTRON_UPSTREAM_README.md docs/ PUSH_INSTRUCTIONS.md
git add notebooks/                                  # calibration_demo.ipynb (and future vio_pilot.ipynb)
git add trajectron/argument_parser.py trajectron/train.py
git add experiments/pedestrians/csv_to_trajectron_data.py
git add experiments/pedestrians/prepare_finetune_data.py 2>/dev/null || true
git add experiments/pedestrians/*.ipynb              # includes presentation_pipeline_results.ipynb
git add experiments/pedestrians/config/ 2>/dev/null || true
git add experiments/pedestrians/compare_v6_v8_fair.py
git add experiments/pedestrians/eval_v7_*.py
git add experiments/pedestrians/visualize_v8.py
git add experiments/pedestrians/example_detections.csv
git add *.ipynb                                      # root-level ranking notebooks
git add experiments/pedestrians/models/eth_attention_radius_3 experiments/pedestrians/models/eth_vel
git add experiments/pedestrians/models/hotel_* experiments/pedestrians/models/univ_* 2>/dev/null || true
git add experiments/pedestrians/models/zara1_* experiments/pedestrians/models/zara2_* 2>/dev/null || true

# kontrol — büyük dosya gelmiş mi?
git status --short
git diff --cached --stat | tail -20
```

> **Önemli:** Eğer `git status` çok büyük model dosyası (`*.pt`, fine-tuned `models_*_uav_*`) gösteriyorsa, **commit etme**, `.gitignore`'u kontrol et. Modellerin büyük olanlarını şimdilik dahil etme (sonra GitHub Releases'a koyarız).

---

## 3. Commit

```bash
git commit -m "Thesis baseline: ETH+UAV fine-tuned models, ranking, VIO plan

- New thesis-focused README (upstream Trajectron++ README moved to TRAJECTRON_UPSTREAM_README.md)
- docs/NOTEBOOKS_GUIDE.md: short narrative for each experiment notebook
- docs/VIO_RESEARCH.md: motivation, theory, validation plan
- docs/PIPELINE.md: module architecture (perception / VIO / forecasting / hand)
- docs/PROGRESS.md: chronological log of decisions and experiments
- .gitignore tightened for videos, weights, large checkpoints
- Notebooks: presentation pipeline results, calibration demo (homography),
  prediction analysis, ranking experiments
"
```

---

## 4. Remote ekle ve push

```bash
# Stanford'un remote'u zaten 'origin' olarak duruyor; onu yedek olarak yeniden adlandır
git remote rename origin upstream

# kendi reponun adresini ekle
git remote add origin <URL>     # ← yukarıda kopyaladığın adres

# kontrol
git remote -v
# upstream  https://github.com/StanfordASL/Trajectron-plus-plus.git (fetch)
# upstream  https://github.com/StanfordASL/Trajectron-plus-plus.git (push)
# origin    https://github.com/<KULLANICI_ADIN>/uav-human-trajectory-prediction.git (fetch)
# origin    https://github.com/<KULLANICI_ADIN>/uav-human-trajectory-prediction.git (push)

# pushla — ilk push olduğu için -u ile upstream da kurarız
git push -u origin thesis/main
```

GitHub seni auth için isteyebilir. Mac'te zaten Cursor / GitHub Desktop / SSH ile ayarlıysan, sorun olmaz. Olmazsa **Personal Access Token** üret (Settings → Developer settings → Personal access tokens → Tokens classic → `repo` scope).

---

## 5. (İsteğe bağlı) Default branch'i `main` yap

GitHub'da reponun ayarlarından:
- `Settings → Branches → Default branch` → `thesis/main` seç → switch.

Veya yerel olarak:

```bash
git branch -m thesis/main main
git push -u origin main
# GitHub'da Settings → Branches → main as default
git push origin --delete thesis/main 2>/dev/null
```

---

## 6. `~/Desktop/uav_perception/` ne olacak?

Şu an worktree dışında ve büyük (videolar, model çıktıları var). En temiz çözüm:

### Yaklaşım 1: Bir alt klasör olarak repoya ekle

```bash
cd <reponun lokal kopyası>     # yani worktree değil de ana klasör
mkdir -p perception
# sadece kod ve gerekli küçük dosyaları kopyala (videoları ve büyük .pt dosyalarını DEĞİL)
cp ~/Desktop/uav_perception/*.py perception/
cp -r ~/Desktop/uav_perception/calibration_output/calibration_demo.ipynb perception/ 2>/dev/null
# .gitignore otomatik olarak .mp4 ve .pt'leri dışlayacak

cd perception/
ls           # sanity check — sadece .py ve .ipynb görmelisin
cd ..
git add perception/
git commit -m "Add perception module (YOLO + calibration scripts)"
git push
```

### Yaklaşım 2: Ayrı bir repo

`uav_perception` adında ayrı bir repo aç ve onu submodule veya bağımsız olarak yönet. Bu tez için fazla overhead, **yaklaşım 1'i öneririm**.

---

## 7. Toplantı öncesi son kontrol

```bash
# GitHub sayfanı aç
open https://github.com/<KULLANICI_ADIN>/uav-human-trajectory-prediction
```

Sayfada şunu görmelisin:
- ✅ README.md güzel görünüyor (başlık, mimari diyagramı, status tablosu)
- ✅ `docs/` klasörü altında 3 dosya: `VIO_RESEARCH.md`, `PIPELINE.md`, `PROGRESS.md`
- ✅ `TRAJECTRON_UPSTREAM_README.md` ayrıca duruyor
- ✅ `.gitignore` var
- ✅ Notebook'lar var
- ✅ Büyük model dosyaları yok

Hocaya **bu URL'yi** verebilirsin.

---

## 8. Toplantıda söyleyeceğin cümle

> "Hocam, GitHub'a koydum — `<URL>`. README'de mimari, `docs/VIO_RESEARCH.md` dosyasında VIO için yaptığım araştırmayı ve validasyon planını, `docs/PROGRESS.md` dosyasında da bugüne kadarki kararları yazdım."

---

## Sorun çıkarsa

| Hata | Çözüm |
|---|---|
| `Permission denied (publickey)` | HTTPS yerine SSH kullanıyorsan, SSH key'ini GitHub'a ekle. Veya `git remote set-url origin https://...` ile HTTPS'e geç ve PAT kullan. |
| `! [rejected] (fetch first)` | İlk push'ta olmaması lazım; eğer GitHub'da otomatik README oluşturdu ise: `git pull --rebase origin main` sonra tekrar push. |
| Dosya çok büyük (>100 MB) | `.gitignore`'u kontrol et, `git rm --cached <dosya>` ile çıkar, tekrar commit + push. |
| `non-fast-forward` | `git pull --rebase origin main && git push` |

Yardım gerekirse, hatanın tam metnini bana yapıştır.
