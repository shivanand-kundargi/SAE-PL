#!/bin/bash
# Download all datasets for MaPLe / CoOp prompt learning
# Usage: bash scripts/download_datasets.sh
# Requires: wget, gdown (pip install gdown), unzip, tar

set -euo pipefail

DATA="/p/lustre5/kundargi1/multimodal-prompt-learning/data"
mkdir -p "$DATA"
echo "Downloading datasets to: $DATA"

require_kaggle_auth() {
    if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
        return 0
    fi

    local kaggle_dir="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
    local kaggle_json="$kaggle_dir/kaggle.json"

    if [ -f "$kaggle_json" ] && grep -q '"username"' "$kaggle_json" && grep -q '"key"' "$kaggle_json"; then
        return 0
    fi

    cat <<EOF
[ERROR] Kaggle credentials not configured.

StanfordCars is downloaded with the Kaggle CLI, which requires either:
  1. Environment variables:
     export KAGGLE_USERNAME="your_username"
     export KAGGLE_KEY="your_api_key"
  2. A JSON file at:
     $kaggle_json

Create the file with:
  mkdir -p "$kaggle_dir"
  chmod 700 "$kaggle_dir"
  cat > "$kaggle_json" <<'JSON'
  {"username":"shivanandkundargi","key":"KGAT_ff6b7e6d71a029695dfe5b187560fd4f"}
  JSON
  chmod 600 "$kaggle_json"

You can create an API token from https://www.kaggle.com/settings.
EOF
    exit 1
}

# Helper: download from Google Drive
gdrive_download() {
    local file_id="$1"
    local output="$2"
    gdown --fuzzy "https://drive.google.com/uc?id=${file_id}" -O "$output"
}

###############################################################################
# 1. Caltech101
###############################################################################
echo "========== Caltech101 =========="
mkdir -p "$DATA/caltech-101"
cd "$DATA/caltech-101"
if [ ! -d "101_ObjectCategories" ]; then
    wget -c https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip -O caltech-101.zip
    unzip -o caltech-101.zip
    # The zip extracts to caltech-101/101_ObjectCategories; move contents up
    if [ -d "caltech-101/101_ObjectCategories" ]; then
        mv caltech-101/101_ObjectCategories .
        rm -rf caltech-101
    fi
    rm -f caltech-101.zip
fi
[ -f split_zhou_Caltech101.json ] || gdrive_download "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN" split_zhou_Caltech101.json

###############################################################################
# 2. OxfordPets
###############################################################################
echo "========== OxfordPets =========="
mkdir -p "$DATA/oxford_pets"
cd "$DATA/oxford_pets"
if [ ! -d "images" ]; then
    wget -c https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    tar xzf images.tar.gz
    rm -f images.tar.gz
fi
if [ ! -d "annotations" ]; then
    wget -c https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    tar xzf annotations.tar.gz
    rm -f annotations.tar.gz
fi
[ -f split_zhou_OxfordPets.json ] || gdrive_download "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs" split_zhou_OxfordPets.json

###############################################################################
# 3. StanfordCars
###############################################################################
echo "========== StanfordCars =========="
mkdir -p "$DATA/stanford_cars"
cd "$DATA/stanford_cars"
# Original Stanford URLs are dead. Use Kaggle CLI.
if [ ! -d "cars_train" ]; then
    require_kaggle_auth
    kaggle datasets download -d eduardo4jesus/stanford-cars-dataset -p "$DATA/stanford_cars" --unzip
fi
[ -f split_zhou_StanfordCars.json ] || gdrive_download "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT" split_zhou_StanfordCars.json

###############################################################################
# 4. Flowers102
###############################################################################
echo "========== Flowers102 =========="
mkdir -p "$DATA/oxford_flowers"
cd "$DATA/oxford_flowers"
if [ ! -d "jpg" ]; then
    wget -c https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    tar xzf 102flowers.tgz
    rm -f 102flowers.tgz
fi
[ -f imagelabels.mat ] || wget -c https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
[ -f cat_to_name.json ] || gdrive_download "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0" cat_to_name.json
[ -f split_zhou_OxfordFlowers.json ] || gdrive_download "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT" split_zhou_OxfordFlowers.json

###############################################################################
# 5. Food101
###############################################################################
echo "========== Food101 =========="
cd "$DATA"
if [ ! -d "food-101" ]; then
    wget -c http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    tar xzf food-101.tar.gz
    rm -f food-101.tar.gz
fi
cd "$DATA/food-101"
[ -f split_zhou_Food101.json ] || gdrive_download "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl" split_zhou_Food101.json

###############################################################################
# 6. FGVCAircraft
###############################################################################
echo "========== FGVCAircraft =========="
cd "$DATA"
if [ ! -d "fgvc_aircraft" ]; then
    wget -c https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
    tar xzf fgvc-aircraft-2013b.tar.gz
    mv fgvc-aircraft-2013b/data fgvc_aircraft
    rm -rf fgvc-aircraft-2013b fgvc-aircraft-2013b.tar.gz
fi

###############################################################################
# 7. SUN397
###############################################################################
echo "========== SUN397 =========="
mkdir -p "$DATA/sun397"
cd "$DATA/sun397"
# Torchvision's SUN397 downloader has historically relied on upstream URLs that
# may go down (e.g., Princeton hosting). Prefer a stable mirror and allow users
# to provide their own archive.
#
# Options:
#   - If SUN397.tar.gz already exists in $DATA/sun397, it will be extracted.
#   - If $SUN397_ARCHIVE is set to a local path, it will be copied and extracted.
#   - Otherwise, download SUN397.tar.gz from Hugging Face (zhengli97/SUN397_Dataset).
if [ ! -d "SUN397" ]; then
    if [ -n "${SUN397_ARCHIVE:-}" ] && [ -f "${SUN397_ARCHIVE}" ]; then
        echo "Using SUN397 archive from: $SUN397_ARCHIVE"
        cp -f "${SUN397_ARCHIVE}" ./SUN397.tar.gz
    fi

    if [ ! -f "SUN397.tar.gz" ]; then
        echo "Downloading SUN397.tar.gz (large) from Hugging Face mirror..."
        if command -v huggingface-cli >/dev/null 2>&1; then
            # `huggingface-cli download` supports resuming; avoids git-lfs.
            # Users can set: export HF_ENDPOINT=https://hf-mirror.com
            huggingface-cli download zhengli97/SUN397_Dataset SUN397.tar.gz \
                --local-dir . --local-dir-use-symlinks False || true
        fi

        if [ ! -f "SUN397.tar.gz" ]; then
            # Fallback to python API if the CLI is unavailable.
            python3 -c "import sys; \
from pathlib import Path; \
try: \
  from huggingface_hub import hf_hub_download; \
except Exception as e: \
  print('[ERROR] huggingface_hub is not installed and huggingface-cli was not found.'); \
  print('Install one of: `pip install huggingface_hub` or `pip install -U huggingface_hub[cli]`.'); \
  sys.exit(2); \
out = hf_hub_download(repo_id='zhengli97/SUN397_Dataset', filename='SUN397.tar.gz', local_dir='.', local_dir_use_symlinks=False); \
p = Path(out); \
if p.name != 'SUN397.tar.gz': \
  p.replace(Path('.') / 'SUN397.tar.gz')" || true
        fi
    fi

    if [ ! -f "SUN397.tar.gz" ]; then
        cat <<EOF
[ERROR] Failed to obtain SUN397.tar.gz.

Tried:
  - Using \$SUN397_ARCHIVE (local path), if set
  - Hugging Face mirror: zhengli97/SUN397_Dataset (via huggingface-cli or huggingface_hub)

Next steps:
  - Set a mirror endpoint if needed:
      export HF_ENDPOINT=https://hf-mirror.com
  - Or download the archive manually and point to it:
      export SUN397_ARCHIVE=/path/to/SUN397.tar.gz
      bash scripts/download_datasets.sh
EOF
        exit 1
    fi

    echo "Extracting SUN397.tar.gz..."
    tar xzf SUN397.tar.gz
fi
[ -f split_zhou_SUN397.json ] || gdrive_download "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq" split_zhou_SUN397.json

###############################################################################
# 8. DTD
###############################################################################
echo "========== DTD =========="
cd "$DATA"
if [ ! -d "dtd" ]; then
    wget -c https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    tar xzf dtd-r1.0.1.tar.gz
    rm -f dtd-r1.0.1.tar.gz
fi
cd "$DATA/dtd"
[ -f split_zhou_DescribableTextures.json ] || gdrive_download "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x" split_zhou_DescribableTextures.json

###############################################################################
# 9. EuroSAT
###############################################################################
echo "========== EuroSAT =========="
mkdir -p "$DATA/eurosat"
cd "$DATA/eurosat"
if [ ! -d "2750" ]; then
    wget -c https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip -O EuroSAT_RGB.zip
    unzip -o EuroSAT_RGB.zip
    # The zip may extract to EuroSAT_RGB/; rename to 2750 if needed
    if [ -d "EuroSAT_RGB" ] && [ ! -d "2750" ]; then
        mv EuroSAT_RGB 2750
    fi
    rm -f EuroSAT_RGB.zip
fi
[ -f split_zhou_EuroSAT.json ] || gdrive_download "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o" split_zhou_EuroSAT.json

###############################################################################
# 10. UCF101
###############################################################################
echo "========== UCF101 =========="
mkdir -p "$DATA/ucf101"
cd "$DATA/ucf101"
[ -d "UCF-101-midframes" ] || gdrive_download "10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O" UCF-101-midframes.zip
if [ -f "UCF-101-midframes.zip" ]; then
    unzip -o UCF-101-midframes.zip
    rm -f UCF-101-midframes.zip
fi
[ -f split_zhou_UCF101.json ] || gdrive_download "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y" split_zhou_UCF101.json

###############################################################################
# 11. ImageNet
###############################################################################
echo "========== ImageNet =========="
mkdir -p "$DATA/imagenet/images"
cd "$DATA/imagenet"
[ -f classnames.txt ] || gdrive_download "1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF" classnames.txt
# Download training images (full 138GB set)
# NOTE: Requires image-net.org login. If wget fails with 403, download manually
# and place ILSVRC2012_img_train.tar in $DATA/imagenet/images/
if [ ! -d "images/train" ]; then
    cd "$DATA/imagenet/images"
    if [ ! -f "ILSVRC2012_img_train.tar" ]; then
        wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar || \
            echo "[WARN] ImageNet train download failed — requires login cookies. Download manually."
    fi
    if [ -f "ILSVRC2012_img_train.tar" ]; then
        mkdir -p train && cd train
        tar xf ../ILSVRC2012_img_train.tar
        # Each class is a nested tar, extract them all
        for f in *.tar; do
            d="${f%.tar}"
            mkdir -p "$d"
            tar xf "$f" -C "$d"
            rm -f "$f"
        done
        cd "$DATA/imagenet/images"
        rm -f ILSVRC2012_img_train.tar
    fi
fi
# Download validation images
if [ ! -d "images/val" ]; then
    cd "$DATA/imagenet/images"
    if [ ! -f "ILSVRC2012_img_val.tar" ]; then
        wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar || \
            echo "[WARN] ImageNet val download failed — requires login cookies. Download manually."
    fi
    if [ -f "ILSVRC2012_img_val.tar" ]; then
        mkdir -p val
        tar xf ILSVRC2012_img_val.tar -C val
        rm -f ILSVRC2012_img_val.tar
    fi
fi
cd "$DATA/imagenet"

###############################################################################
# 12. ImageNetV2
###############################################################################
echo "========== ImageNetV2 =========="
mkdir -p "$DATA/imagenetv2"
cd "$DATA/imagenetv2"
if [ ! -d "imagenetv2-matched-frequency-format-val" ]; then
    TAR="imagenetv2-matched-frequency.tar.gz"
    URL="https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/${TAR}?download=true"

    # If a previous attempt left a partial/HTML file, remove it.
    # Note: despite the .tar.gz suffix, the upstream file may be an uncompressed tar.
    if [ -f "$TAR" ] && ! tar tf "$TAR" >/dev/null 2>&1 && ! tar tzf "$TAR" >/dev/null 2>&1; then
        echo "[WARN] $TAR exists but is not a valid tar/tar.gz; deleting and re-downloading."
        rm -f "$TAR"
    fi

    if [ ! -f "$TAR" ]; then
        echo "Downloading $TAR from Hugging Face..."

        # Prefer Hugging Face tools to avoid brittle redirect/range behavior (e.g., 416 on xet/cas URLs).
        if command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli download vaishaal/ImageNetV2 "$TAR" \
                --local-dir . --local-dir-use-symlinks False || true
        fi

        if [ ! -f "$TAR" ]; then
            python3 -c "import sys; \
try: \
  from huggingface_hub import hf_hub_download; \
except Exception: \
  sys.exit(2); \
hf_hub_download(repo_id='vaishaal/ImageNetV2', repo_type='dataset', filename='imagenetv2-matched-frequency.tar.gz', local_dir='.', local_dir_use_symlinks=False)" || true
        fi

        if [ ! -f "$TAR" ]; then
            # Last resort: plain HTTP download (no resume) with content-disposition.
            wget --content-disposition -O "$TAR" "$URL"
        fi
    fi

    echo "Extracting $TAR..."
    if tar tzf "$TAR" >/dev/null 2>&1; then
        tar xzf "$TAR"
    elif tar tf "$TAR" >/dev/null 2>&1; then
        tar xf "$TAR"
    else
        echo "[ERROR] $TAR is not a valid tar or tar.gz archive."
        exit 1
    fi
    rm -f "$TAR"
fi
[ -f classnames.txt ] || cp "$DATA/imagenet/classnames.txt" . 2>/dev/null || echo "[SKIP] Copy classnames.txt after ImageNet is ready"

###############################################################################
# 13. ImageNet-Sketch

echo "========== ImageNet-Sketch =========="
mkdir -p "$DATA/imagenet-sketch"
cd "$DATA/imagenet-sketch"
if [ ! -d "images" ]; then
    # From the official repo: https://github.com/HaohanWang/ImageNet-Sketch
    # ~7.8 GB compressed, ~8.4 GB extracted
    gdrive_download "1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA" ImageNet-Sketch.zip
    unzip -o ImageNet-Sketch.zip
    # Rename extracted folder to 'images' if needed
    if [ -d "sketch" ] && [ ! -d "images" ]; then
        mv sketch images
    fi
    rm -f ImageNet-Sketch.zip
fi
[ -f classnames.txt ] || cp "$DATA/imagenet/classnames.txt" . 2>/dev/null || echo "[SKIP] Copy classnames.txt after ImageNet is ready"

###############################################################################
# 14. ImageNet-A
###############################################################################
echo "========== ImageNet-A =========="
mkdir -p "$DATA/imagenet-adversarial"
cd "$DATA/imagenet-adversarial"
if [ ! -d "imagenet-a" ]; then
    wget -c https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
    tar xf imagenet-a.tar
    rm -f imagenet-a.tar
fi
[ -f classnames.txt ] || cp "$DATA/imagenet/classnames.txt" . 2>/dev/null || echo "[SKIP] Copy classnames.txt after ImageNet is ready"

###############################################################################
# 15. ImageNet-R
###############################################################################
echo "========== ImageNet-R =========="
mkdir -p "$DATA/imagenet-rendition"
cd "$DATA/imagenet-rendition"
if [ ! -d "imagenet-r" ]; then
    wget -c https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
    tar xf imagenet-r.tar
    rm -f imagenet-r.tar
fi
[ -f classnames.txt ] || cp "$DATA/imagenet/classnames.txt" . 2>/dev/null || echo "[SKIP] Copy classnames.txt after ImageNet is ready"

###############################################################################
echo ""
echo "========== DONE =========="
echo "All automated downloads complete."
echo ""
echo "If any downloads failed, check:"
echo "  - ImageNet: requires image-net.org login cookies for wget (download manually if 403)"
echo "  - StanfordCars: requires kaggle CLI + API token (pip install kaggle)"
echo "  - SUN397: requires torchvision (pip install torchvision)"
echo "  - Google Drive files: requires gdown (pip install gdown)"
echo ""
echo "Note: Some Google Drive downloads may hit rate limits. Re-run the script to resume."
