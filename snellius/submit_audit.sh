#!/usr/bin/env bash
# Perturbation audit of every exp125 arm (no training, a few minutes each): what a block can see.
#   bash snellius/submit_audit.sh        then:  grep -h "\[audit\]" exp125-*.out | grep -v " ok "
C=/scratch-shared/$USER/hspec_code
sub() { CODE="$1" NAME="audit_$2" EPOCHS=0 OUT=/scratch-shared/$USER/hspec_code/_audit EXTRA="--audit ${3:-}" \
        sbatch --time=00:30:00 --export=ALL snellius/job_exp125.sbatch; }
sub 0               tokens
sub real            real
sub 64              code64
sub state:29        state29
sub path:0,1,27,28  path4       "--block-chunk 8"
sub path:0-3        first4_lora "--lora-rank 64 --block-chunk 8"
sub path:0,1,21-28a path10a     "--block-chunk 8"
sub self            self64      "--from-ckpt $C/r64"
sub xattn:29        xattn29
