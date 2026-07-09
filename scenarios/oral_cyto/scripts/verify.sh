#!/bin/bash
# Comprehensive verification suite for the oral_cyto scenario.
#
# Runs static parse checks, cross-config coherence, file hygiene, doc
# completeness, git state, container sanity, functional smoke tests, and
# performance sanity. Exit 0 on no-FAIL, 1 if any FAIL. WARNs are non-blocking.
#
# Flags:
#   --fast      Skip docker-based sections (6, 7, 8). Static checks only.
#   --no-train  Skip the training smoke test (7.5) but keep build + preprocess.
#   -v          Show underlying command output.

set -uo pipefail

FAST=0
NO_TRAIN=0
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) FAST=1 ;;
        --no-train) NO_TRAIN=1 ;;
        -v|--verbose) VERBOSE=1 ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$REPO_ROOT"
SCENARIO_DIR="scenarios/oral_cyto"

PASS=0; FAIL=0; WARN=0; SKIP_N=0
FAIL_LIST=()

pass() { PASS=$((PASS+1)); echo "[PASS] $1"; }
fail() { FAIL=$((FAIL+1)); FAIL_LIST+=("$1"); echo "[FAIL] $1 -> $2"; }
warn() { WARN=$((WARN+1)); echo "[WARN] $1 -> $2"; }
skip() { SKIP_N=$((SKIP_N+1)); echo "[SKIP] $1 -> $2"; }
banner() {
    echo ""
    echo "============================================================"
    echo "  Section $1: $2"
    echo "============================================================"
}
vlog() { if [[ "$VERBOSE" == "1" ]]; then echo "  | $*"; fi; }
vcat() { if [[ "$VERBOSE" == "1" ]] && [[ -s "$1" ]]; then sed 's/^/  | /' "$1"; fi; }

# ============================================================
# Section 1: Static parse correctness
# ============================================================
banner 1 "Static parse correctness"

# 1.1 py_compile
ERR=""
for f in $(find "$SCENARIO_DIR/src" -name "*.py" 2>/dev/null); do
    if ! python3 -m py_compile "$f" 2>/tmp/_v_err; then
        ERR="$ERR $f"
        vcat /tmp/_v_err
    fi
done
if [[ -z "$ERR" ]]; then pass "1.1 py_compile"; else fail "1.1 py_compile" "Failed:$ERR"; fi

# 1.2 bash -n
# Note: export-variables.sh is excluded — by framework convention it holds
# placeholder values like <azure-location> that fail bash -n until filled
# in. See CLAUDE.md Known quirks.
ERR=""
for f in $(find "$SCENARIO_DIR" -name "*.sh" -not -name "export-variables.sh" 2>/dev/null); do
    if ! bash -n "$f" 2>/tmp/_v_err; then
        ERR="$ERR $f"
        vcat /tmp/_v_err
    fi
done
if [[ -z "$ERR" ]]; then pass "1.2 bash -n (excl. export-variables.sh)"; else fail "1.2 bash -n" "Failed:$ERR"; fi

# 1.3 jq empty on JSONs in config/policy/contract
ERR=""
for f in $(find "$SCENARIO_DIR/config" "$SCENARIO_DIR/policy" "$SCENARIO_DIR/contract" -name "*.json" 2>/dev/null); do
    if ! jq empty "$f" 2>/tmp/_v_err; then
        ERR="$ERR $f"
        vcat /tmp/_v_err
    fi
done
if [[ -z "$ERR" ]]; then pass "1.3 jq empty on JSONs"; else fail "1.3 jq empty" "Invalid:$ERR"; fi

# 1.4 yaml.safe_load on all .yml/.yaml under deployment/
ERR=""
for f in $(find "$SCENARIO_DIR/deployment" \( -name "*.yml" -o -name "*.yaml" \) 2>/dev/null); do
    if ! python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$f" 2>/tmp/_v_err; then
        ERR="$ERR $f"
        vcat /tmp/_v_err
    fi
done
if [[ -z "$ERR" ]]; then pass "1.4 yaml.safe_load"; else fail "1.4 yaml.safe_load" "Invalid:$ERR"; fi

# 1.5 Dockerfile FROM-line sanity
ERR=""
for f in $(find "$SCENARIO_DIR/ci" -name "Dockerfile.*" 2>/dev/null); do
    if ! head -1 "$f" | grep -qE "^FROM [a-zA-Z0-9./:_-]+"; then
        ERR="$ERR $f"
    fi
done
if [[ -z "$ERR" ]]; then pass "1.5 Dockerfile FROM line"; else fail "1.5 Dockerfile FROM" "Bad first line:$ERR"; fi

# ============================================================
# Section 2: Cross-config coherence
# ============================================================
banner 2 "Cross-config coherence"

# 2.1 No leftover oral_cyto_C/D/E references (excludes this script's own pattern strings)
hits=$(grep -rE "oral_cyto_[CDE]|oralcyto[cde]|ORAL_CYTO_[CDE]" "$SCENARIO_DIR" --exclude-dir=scripts 2>/dev/null || true)
if [[ -z "$hits" ]]; then pass "2.1 no leftover C/D/E TDP refs"; else fail "2.1 leftover TDP refs" "Found: $(echo "$hits" | head -3 | tr '\n' '|')"; fi

# 2.2 Image tag consistency: only canonical 3 names
refs=$(grep -rh -oE "preprocess-oral-cyto-[a-z0-9]+|preprocess-oralcyto[a-z0-9]+|oral-cyto-model-save|oral-cyto-modelsave" \
       "$SCENARIO_DIR/ci/" "$SCENARIO_DIR/deployment/local/" 2>/dev/null | sort -u)
canonical=$'oral-cyto-model-save\npreprocess-oral-cyto-a\npreprocess-oral-cyto-b'
if [[ "$refs" == "$canonical" ]]; then
    pass "2.2 image tag consistency"
else
    fail "2.2 image tags" "Got: $(echo "$refs" | tr '\n' ',') | want: preprocess-oral-cyto-{a,b}, oral-cyto-model-save"
fi

# 2.3 Mount path consistency (regex allows uppercase A/B in TDP suffix)
hits=$(grep -rh -oE "/mnt/remote/[A-Za-z_]+" "$SCENARIO_DIR/config" "$SCENARIO_DIR/deployment/local" 2>/dev/null | sort -u)
# Expected only: /mnt/remote/oral_cyto_A, /mnt/remote/oral_cyto_B, /mnt/remote/model, /mnt/remote/output, /mnt/remote/config
bad=$(echo "$hits" | grep -v -E "^/mnt/remote/(oral_cyto_[AB]|model|output|config)$" || true)
if [[ -z "$bad" ]]; then pass "2.3 mount path consistency"; else fail "2.3 mount paths" "Unexpected mounts: $(echo "$bad" | tr '\n' ',')"; fi

# 2.4 Privacy invariant (policy.rego XOR rule)
ip_tpl=$(jq -r '.config.is_private' "$SCENARIO_DIR/config/templates/train_config_template.json" 2>/dev/null)
priv_n=$(jq -r '.constraints[0].privacy | length' "$SCENARIO_DIR/contract/contract.json" 2>/dev/null)
ip_pc=$(jq -r '.pipeline[1].config.is_private' "$SCENARIO_DIR/config/pipeline_config.json" 2>/dev/null)
if [[ "$ip_tpl" == "false" && "$priv_n" == "0" && "$ip_pc" == "false" ]]; then
    pass "2.4 privacy invariant (DP-off, no contract bounds)"
else
    fail "2.4 privacy invariant" "tpl.is_private=$ip_tpl, contract.privacy len=$priv_n, pipeline[1].is_private=$ip_pc"
fi

# 2.5 consolidate_pipeline.sh idempotency
cp "$SCENARIO_DIR/config/pipeline_config.json" /tmp/_v_pc_before.json
( cd "$SCENARIO_DIR" && ./config/consolidate_pipeline.sh >/tmp/_v_consolidate 2>&1 )
rc=$?
vcat /tmp/_v_consolidate
if [[ $rc -ne 0 ]]; then
    fail "2.5 consolidate idempotency" "consolidate_pipeline.sh returned $rc"
elif diff -q /tmp/_v_pc_before.json "$SCENARIO_DIR/config/pipeline_config.json" >/dev/null; then
    pass "2.5 consolidate idempotency"
else
    fail "2.5 consolidate idempotency" "pipeline_config.json changed across re-run"
    cp /tmp/_v_pc_before.json "$SCENARIO_DIR/config/pipeline_config.json"
fi

# 2.6 No "brats" in code/config (allowed in .md as documentary; excludes this script)
hits=$(grep -rli "brats" "$SCENARIO_DIR" --include="*.py" --include="*.json" --include="*.sh" --include="*.yml" --include="Dockerfile.*" --exclude-dir=scripts 2>/dev/null || true)
if [[ -z "$hits" ]]; then pass "2.6 no brats refs in code/config"; else fail "2.6 brats refs" "$(echo "$hits" | tr '\n' ',')"; fi

# 2.7 G1 invariant: 7 canonical values (each expr parenthesized — jq's
# `A | B, C` parses as `A | (B, C)` so without parens, C is evaluated
# against the array, not the root)
got=$(jq -r '(.pipeline | length),
             (.pipeline[].name),
             (.pipeline[1].config.is_private),
             (.pipeline[1].config.model_config.layers.in_conv.params.in_channels),
             (.pipeline[1].config.dataset_config.pairing.folder_pattern),
             (.pipeline[0].config.joined_dataset)' \
           "$SCENARIO_DIR/config/pipeline_config.json" 2>/dev/null)
expected=$'2\nDirectoryJoin\nTrain_DL\nfalse\n3\noral_cyto_*\n/tmp/oral_cyto_joined'
if [[ "$got" == "$expected" ]]; then
    pass "2.7 G1 canonical pipeline_config values"
else
    fail "2.7 G1 canonical values" "Got: $(echo "$got" | tr '\n' ',')"
fi

# ============================================================
# Section 3: File hygiene & security
# ============================================================
banner 3 "File hygiene & security"

# 3.1 No tracked data files
hits=$(git ls-files "$SCENARIO_DIR" 2>/dev/null | grep -E '\.(png|geojson|zip|tar\.gz|tgz|mrxs)$' || true)
if [[ -z "$hits" ]]; then pass "3.1 no tracked data files"; else fail "3.1 tracked data files" "$(echo "$hits" | tr '\n' ',')"; fi

# 3.2 Credential scan (excludes this script which contains the regex literals)
hits=$(grep -rEn "KGAT_[A-Za-z0-9]{32}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{40,}|-----BEGIN.*PRIVATE KEY-----" "$SCENARIO_DIR" --exclude-dir=scripts 2>/dev/null || true)
if [[ -z "$hits" ]]; then pass "3.2 no credential patterns"; else fail "3.2 credential patterns" "Found: $(echo "$hits" | head -2 | tr '\n' '|')"; fi

# 3.3 Dev-env path leakage (excludes .md docs and this script's regex literals)
hits=$(grep -rEn "/root/|/home/[a-z]+/|192\.168\.|10\.0\.0\.|127\.0\.0\.1" "$SCENARIO_DIR" --exclude="*.md" --exclude-dir=scripts 2>/dev/null | \
       grep -v "localhost:" || true)
# Filter known-OK patterns: ${VAR:-"https://localhost:..."} fallback defaults
hits=$(echo "$hits" | grep -v -E '\$\{[^}]+:-"https?://localhost' || true)
if [[ -z "$hits" ]]; then pass "3.3 no dev-env path leakage"; else fail "3.3 dev-env paths" "Found: $(echo "$hits" | head -2 | tr '\n' '|')"; fi

# 3.4 Shell scripts executable
nonexec=$(find "$SCENARIO_DIR" -name "*.sh" -not -perm -u+x 2>/dev/null || true)
if [[ -z "$nonexec" ]]; then pass "3.4 shell scripts executable"; else fail "3.4 nonexecutable scripts" "$(echo "$nonexec" | tr '\n' ',')"; fi

# 3.5 No CRLF
crlf=$(find "$SCENARIO_DIR" -type f \( -name "*.sh" -o -name "*.py" \) -exec file {} \; 2>/dev/null | grep -i "CRLF" || true)
if [[ -z "$crlf" ]]; then pass "3.5 no CRLF line endings"; else fail "3.5 CRLF endings" "$(echo "$crlf" | tr '\n' ',')"; fi

# 3.6 No editor leavings
editor=$(find "$SCENARIO_DIR" \( -name "*~" -o -name "*.swp" -o -name ".DS_Store" -o -name "Thumbs.db" \) 2>/dev/null || true)
if [[ -z "$editor" ]]; then pass "3.6 no editor leavings"; else fail "3.6 editor leavings" "$(echo "$editor" | tr '\n' ',')"; fi

# ============================================================
# Section 4: Documentation completeness
# ============================================================
banner 4 "Documentation completeness"

# 4.1 README has required sections
readme="$SCENARIO_DIR/README.md"
required=("Scenario Type" "Scenario Description" "Privacy properties" "Deploy locally" "References")
missing=""
for sec in "${required[@]}"; do
    if ! grep -qE "^## .*$sec" "$readme" 2>/dev/null; then
        missing="$missing|$sec"
    fi
done
if [[ -z "$missing" ]]; then pass "4.1 README required sections"; else fail "4.1 README sections" "Missing:$missing"; fi

# 4.2 CLAUDE.md has Known quirks
if grep -q "## Known quirks" "$SCENARIO_DIR/CLAUDE.md" 2>/dev/null; then
    pass "4.2 CLAUDE.md Known quirks"
else
    fail "4.2 CLAUDE.md Known quirks" "Section missing or CLAUDE.md absent"
fi

# 4.3 All .py have CC0 1.0 header
missing=""
for f in "$SCENARIO_DIR"/src/*.py; do
    if ! grep -q "CC0 1.0 Universal" "$f" 2>/dev/null; then
        missing="$missing $f"
    fi
done
if [[ -z "$missing" ]]; then pass "4.3 CC0 1.0 headers"; else fail "4.3 CC0 headers" "Missing:$missing"; fi

# 4.4 No TODO/FIXME/XXX in code paths
hits=$(grep -rnE "TODO|FIXME|XXX" "$SCENARIO_DIR/src" "$SCENARIO_DIR/config" "$SCENARIO_DIR/ci" "$SCENARIO_DIR/deployment/local" 2>/dev/null || true)
if [[ -z "$hits" ]]; then pass "4.4 no TODO/FIXME/XXX in code"; else fail "4.4 TODO markers" "$(echo "$hits" | head -2 | tr '\n' '|')"; fi

# 4.5 References cite paper + dataset + repo
miss=""
grep -qE "arXiv.*2506\.06990|arxiv\.org/abs/2506\.06990" "$readme" 2>/dev/null || miss="$miss arXiv-2506.06990"
grep -qE "kaggle\.com/datasets/abhijeetptl5/oral-cytology-dataset" "$readme" 2>/dev/null || miss="$miss kaggle-dataset"
grep -qE "github\.com/abhijeetptl5/oral_cyto_dataset" "$readme" 2>/dev/null || miss="$miss github-repo"
if [[ -z "$miss" ]]; then pass "4.5 References citations"; else fail "4.5 References" "Missing:$miss"; fi

# ============================================================
# Section 5: Git hygiene
# ============================================================
banner 5 "Git hygiene"

# 5.1 Current branch
br=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [[ "$br" == "scenario/oral-cyto" ]]; then pass "5.1 branch is scenario/oral-cyto"; else warn "5.1 branch" "On $br (expected scenario/oral-cyto)"; fi

# 5.2 Working tree clean inside scenario
dirty=$(git status --short "$SCENARIO_DIR" 2>/dev/null)
if [[ -z "$dirty" ]]; then pass "5.2 working tree clean in scenario"; else fail "5.2 working tree dirty" "$(echo "$dirty" | tr '\n' '|')"; fi

# 5.3 No tracked >100KB
large=$(git ls-files "$SCENARIO_DIR" 2>/dev/null | while read f; do
    [[ -f "$f" ]] && sz=$(wc -c <"$f") && [[ $sz -gt 102400 ]] && echo "$sz $f"
done)
if [[ -z "$large" ]]; then pass "5.3 no tracked files >100KB"; else fail "5.3 large tracked files" "$(echo "$large" | tr '\n' '|')"; fi

# ============================================================
# Section 6: Containerization
# ============================================================
if [[ "$FAST" == "1" ]]; then
    banner 6 "Containerization (SKIPPED via --fast)"
    skip "6.* docker checks" "--fast"
else
    banner 6 "Containerization"

    # 6.1 framework image exists
    if docker image inspect depa-training:latest >/dev/null 2>&1; then
        pass "6.1 depa-training:latest exists"
    else
        fail "6.1 depa-training:latest" "Image not found locally (build from src/train + ci/Dockerfile.train)"
    fi

    # 6.2 ./ci/build.sh runs (cached, fast on re-run)
    ( cd "$SCENARIO_DIR" && ./ci/build.sh >/tmp/_v_build 2>&1 )
    rc=$?
    vcat /tmp/_v_build
    if [[ $rc -eq 0 ]]; then pass "6.2 ci/build.sh runs"; else fail "6.2 ci/build.sh" "Exit $rc"; fi

    # 6.3 All three scenario images present
    miss=""
    for img in preprocess-oral-cyto-a:latest preprocess-oral-cyto-b:latest oral-cyto-model-save:latest; do
        docker image inspect "$img" >/dev/null 2>&1 || miss="$miss $img"
    done
    if [[ -z "$miss" ]]; then pass "6.3 all scenario images present"; else fail "6.3 missing images" "$miss"; fi

    # 6.4 preprocess containers import deps
    miss=""
    for img in preprocess-oral-cyto-a preprocess-oral-cyto-b; do
        docker run --rm "$img" python3 -c "import torch, cv2, PIL, geopandas, shapely" >/tmp/_v_imp 2>&1 || { miss="$miss $img"; vcat /tmp/_v_imp; }
    done
    if [[ -z "$miss" ]]; then pass "6.4 preprocess imports OK"; else fail "6.4 preprocess imports" "Failed:$miss"; fi

    # 6.5 modelsave container import
    if docker run --rm oral-cyto-model-save python3 -c "import torch, safetensors, packaging" >/tmp/_v_imp 2>&1; then
        pass "6.5 modelsave imports OK"
    else
        vcat /tmp/_v_imp
        fail "6.5 modelsave imports" "Failed"
    fi

    # 6.6 cv2.fillPoly functional
    if docker run --rm preprocess-oral-cyto-a python3 -c \
        "import cv2, numpy as np; m = np.zeros((10,10), np.uint8); cv2.fillPoly(m, [np.array([[1,1],[8,1],[8,8]], np.int32)], 1); assert m.sum() == 36" \
        >/tmp/_v_imp 2>&1; then
        pass "6.6 cv2.fillPoly functional"
    else
        vcat /tmp/_v_imp
        fail "6.6 cv2.fillPoly" "Functional test failed"
    fi
fi

# ============================================================
# Section 7: Functional smoke tests
# ============================================================
if [[ "$FAST" == "1" ]]; then
    banner 7 "Functional smoke tests (SKIPPED via --fast)"
    skip "7.* smoke tests" "--fast"
else
    banner 7 "Functional smoke tests"

    # 7.1 consolidate (already covered by 2.5/2.7); re-affirm pipeline_config exists
    if [[ -s "$SCENARIO_DIR/config/pipeline_config.json" ]]; then
        pass "7.1 pipeline_config.json present + non-empty"
    else
        fail "7.1 pipeline_config.json" "Missing or empty"
    fi

    # 7.2 save-model produces model.safetensors
    ( cd "$SCENARIO_DIR/deployment/local" && ./save-model.sh >/tmp/_v_save 2>&1 )
    rc=$?
    vcat /tmp/_v_save
    safetensors="$SCENARIO_DIR/modeller/models/model.safetensors"
    if [[ $rc -eq 0 ]] && [[ -s "$safetensors" ]]; then
        pass "7.2 save-model produces model.safetensors"
    else
        fail "7.2 save-model" "rc=$rc, file=$([[ -s "$safetensors" ]] && echo "$(wc -c <"$safetensors")B" || echo "missing")"
    fi

    # 7.3 Forward pass functional in modelsave container
    if [[ -s "$safetensors" ]]; then
        docker run --rm \
            -v "$REPO_ROOT/$SCENARIO_DIR/modeller/models:/mnt/model" \
            -v "$REPO_ROOT/$SCENARIO_DIR/config:/mnt/config" \
            -v "$REPO_ROOT/$SCENARIO_DIR/src:/src" \
            oral-cyto-model-save python3 -c "
import sys; sys.path.insert(0, '/src')
import json, torch
from safetensors.torch import load_file
from model_constructor import ModelFactory
with open('/mnt/config/model_config.json') as f: cfg = json.load(f)
model = ModelFactory.load_from_dict(cfg)
state = load_file('/mnt/model/model.safetensors')
model.load_state_dict(state); model.eval()
n_params = sum(p.numel() for p in model.parameters())
assert 1_500_000 <= n_params <= 3_000_000, f'param count {n_params} out of range'
x = torch.randn(1, 3, 256, 256)
with torch.no_grad(): y = model(x)
assert y.shape == (1, 1, 256, 256), f'shape {y.shape}'
assert y.min().item() >= 0 and y.max().item() <= 1, f'range [{y.min().item()}, {y.max().item()}]'
assert not torch.isnan(y).any().item(), 'NaN'
assert not torch.isinf(y).any().item(), 'Inf'
print(f'forward OK params={n_params:,} range=[{y.min().item():.4f},{y.max().item():.4f}]')
" >/tmp/_v_fwd 2>&1
        rc=$?
        vcat /tmp/_v_fwd
        if [[ $rc -eq 0 ]]; then pass "7.3 forward pass functional"; else fail "7.3 forward pass" "$(tail -3 /tmp/_v_fwd | tr '\n' '|')"; fi
    else
        skip "7.3 forward pass" "model.safetensors missing (7.2 failed)"
    fi

    # 7.4 Preprocess (only if raw data present)
    raw_dir="$SCENARIO_DIR/data/raw/oral_cytology_dataset"
    if [[ -d "$raw_dir" ]]; then
        ( cd "$SCENARIO_DIR/deployment/local" && ./preprocess.sh >/tmp/_v_pp 2>&1 )
        rc=$?
        vcat /tmp/_v_pp
        if [[ $rc -ne 0 ]]; then
            fail "7.4 preprocess.sh" "Exit $rc"
        else
            a_count=$(find "$SCENARIO_DIR/data/oral_cyto_A/preprocessed" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            b_count=$(find "$SCENARIO_DIR/data/oral_cyto_B/preprocessed" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            if [[ "$a_count" == "349" ]] && [[ "$b_count" == "390" ]]; then
                # Sample mask validity check
                samp=$(find "$SCENARIO_DIR/data/oral_cyto_A/preprocessed" -name "*_seg.png" | head -1)
                if [[ -n "$samp" ]]; then
                    ok=$(docker run --rm -v "$REPO_ROOT/$samp:/m.png:ro" preprocess-oral-cyto-a python3 -c "
import numpy as np; from PIL import Image
m = np.array(Image.open('/m.png'))
assert m.shape == (256, 256), f'shape {m.shape}'
assert (m > 0).any(), 'mask all zero'
print('mask OK')
" 2>&1)
                    if echo "$ok" | grep -q "mask OK"; then
                        pass "7.4 preprocess counts (A=349, B=390) + mask sample valid"
                    else
                        fail "7.4 preprocess mask" "$ok"
                    fi
                else
                    pass "7.4 preprocess counts (A=349, B=390)"
                fi
            else
                fail "7.4 preprocess counts" "A=$a_count (expected 349), B=$b_count (expected 390)"
            fi
        fi
    else
        skip "7.4 preprocess" "data/raw/oral_cytology_dataset missing — run kaggle download first"
    fi

    # 7.5 Full train smoke test
    if [[ "$NO_TRAIN" == "1" ]]; then
        skip "7.5 train smoke" "--no-train"
    else
        ( cd "$SCENARIO_DIR/deployment/local" && ./train.sh >/tmp/_v_train 2>&1 )
        rc=$?
        vcat /tmp/_v_train
        trained="$SCENARIO_DIR/modeller/output/trained_model.safetensors"
        metrics="$SCENARIO_DIR/modeller/output/evaluation_metrics.json"
        if [[ $rc -ne 0 ]]; then
            fail "7.5 train smoke" "Exit $rc"
        elif [[ ! -s "$trained" ]]; then
            fail "7.5 train output" "trained_model.safetensors missing"
        elif [[ ! -s "$metrics" ]]; then
            fail "7.5 eval metrics" "evaluation_metrics.json missing"
        else
            # Check that required keys are present and values are finite
            finite_ok=$(python3 -c "
import json, math
m = json.load(open('$metrics'))
keys = ['dice_score', 'jaccard_index', 'hausdorff_distance']
miss = [k for k in keys if k not in m]
if miss:
    print(f'missing keys: {miss}')
else:
    nonfin = [k for k in keys if not math.isfinite(m[k])]
    if nonfin:
        print(f'non-finite: {nonfin}')
    else:
        print('OK')
" 2>&1)
            if [[ "$finite_ok" == "OK" ]]; then
                pass "7.5 train smoke test (exit 0, finite metrics)"
            else
                fail "7.5 eval metrics" "$finite_ok"
            fi
        fi
    fi
fi

# ============================================================
# Section 8: Performance sanity (WARN, not FAIL)
# ============================================================
if [[ "$FAST" == "1" ]]; then
    banner 8 "Performance sanity (SKIPPED via --fast)"
    skip "8.* perf" "--fast"
else
    banner 8 "Performance sanity"

    # 8.1 Image sizes
    for img in preprocess-oral-cyto-a preprocess-oral-cyto-b; do
        sz=$(docker image inspect --format "{{.Size}}" "$img:latest" 2>/dev/null || echo 0)
        mb=$((sz / 1024 / 1024))
        if [[ $sz -lt 734003200 ]] || [[ $sz -gt 2147483648 ]]; then
            warn "8.1 $img size" "${mb}MB outside 700MB-2GB"
        else
            pass "8.1 $img size (${mb}MB)"
        fi
    done
    sz=$(docker image inspect --format "{{.Size}}" oral-cyto-model-save:latest 2>/dev/null || echo 0)
    mb=$((sz / 1024 / 1024))
    if [[ $sz -lt 419430400 ]] || [[ $sz -gt 1610612736 ]]; then
        warn "8.1 oral-cyto-model-save size" "${mb}MB outside 400MB-1.5GB"
    else
        pass "8.1 oral-cyto-model-save size (${mb}MB)"
    fi

    # 8.2 Param count (parse safetensors header)
    safetensors="$SCENARIO_DIR/modeller/models/model.safetensors"
    if [[ -s "$safetensors" ]]; then
        n=$(python3 -c "
import struct, json
from functools import reduce
with open('$safetensors', 'rb') as f:
    hl = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(hl))
print(sum(reduce(lambda a,b: a*b, v['shape'], 1) for k,v in h.items() if k != '__metadata__'))
" 2>&1)
        if [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -ge 1500000 ]] && [[ "$n" -le 3000000 ]]; then
            pass "8.2 model params ($n)"
        else
            warn "8.2 model params" "$n outside [1.5M, 3.0M]"
        fi
    else
        skip "8.2 model params" "model.safetensors missing"
    fi

    # 8.3 Preprocessed footprint
    if [[ -d "$SCENARIO_DIR/data/oral_cyto_A/preprocessed" ]] && [[ -d "$SCENARIO_DIR/data/oral_cyto_B/preprocessed" ]]; then
        total=$(du -sb "$SCENARIO_DIR/data/oral_cyto_A/preprocessed" "$SCENARIO_DIR/data/oral_cyto_B/preprocessed" 2>/dev/null | awk '{s+=$1} END {print s}')
        mb=$((total / 1024 / 1024))
        if [[ $total -lt 209715200 ]]; then
            pass "8.3 preprocessed footprint (${mb}MB)"
        else
            warn "8.3 preprocessed footprint" "${mb}MB exceeds 200MB"
        fi
    else
        skip "8.3 preprocessed footprint" "no preprocessed/ dirs"
    fi
fi

# ============================================================
# Summary
# ============================================================
banner "SUMMARY" "results"
total=$((PASS + FAIL + WARN + SKIP_N))
echo "  Total: $total | PASS: $PASS | FAIL: $FAIL | WARN: $WARN | SKIP: $SKIP_N"
if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed checks:"
    for f in "${FAIL_LIST[@]}"; do
        echo "    - $f"
    done
fi

if [[ $FAIL -gt 0 ]]; then exit 1; else exit 0; fi
