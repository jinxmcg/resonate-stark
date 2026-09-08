#!/bin/bash
# Provision a fresh vast 5090 with the prime tree: code, the data a run needs, HF cache, deps.
# Usage: bootstrap.sh <ip> <port>
IP=$1; PORT=$2; P="/mnt/geocore/geocore/prime"
ssh -p $PORT -o StrictHostKeyChecking=accept-new root@$IP "mkdir -p /workspace/prime/data/oof /workspace/prime/models /workspace/prime/logs /workspace/.hf_home/hub"
rsync -az -e "ssh -p $PORT" --exclude ".venv" --exclude ".git" --exclude "__pycache__" \
  $P/*.py $P/pyproject.toml $P/lib $P/scripts root@$IP:/workspace/prime/
rsync -a -e "ssh -p $PORT" $P/data/kg.npz $P/data/names.json $P/data/dicts.json $P/data/signatures.json \
  $P/data/docs.jsonl $P/data/doc_emb_bgeft2.npy $P/data/doc_emb_bge.npy $P/data/empty.json \
  $P/data/lp_labels.json $P/data/para_train.json $P/data/para_train_terse.json $P/data/para_val.json \
  $P/data/llmparse_train.json $P/data/llmparse_train_para.json $P/data/fusion_bgeft2.json \
  $P/data/lparse_train_oof.json $P/data/lparse_train_oof_para.json $P/data/lparse_val.json $P/data/lparse_val_para.json \
  $P/data/text_train_bgeft2oof.json $P/data/text_train_bgeft2oof_para.json $P/data/text_val_bgeft2.json $P/data/text_val_bgeft2_para.json \
  $P/data/text_train_pjointoof.json $P/data/text_train_pjointoof_para.json $P/data/text_val_pjoint.json $P/data/text_val_pjoint_para.json \
  $P/data/rerank_lp_ancf_bgeft2_pjoint_aug_oof.json root@$IP:/workspace/prime/data/
rsync -a -e "ssh -p $PORT" $P/data/oof/bgeft2_f0.json $P/data/oof/p_joint_f0.json $P/data/oof/lparse_train_f0.json root@$IP:/workspace/prime/data/oof/
rsync -a -e "ssh -p $PORT" $P/models/p_k12b4_50k.pt $P/models/p_joint.pt $P/models/p_joint_head.pt $P/models/lp_p3.pt root@$IP:/workspace/prime/models/
rsync -a -e "ssh -p $PORT" $P/models/bge_ft2 $P/models/p_joint_enc root@$IP:/workspace/prime/models/
rsync -a -e "ssh -p $PORT" /root/.cache/huggingface/hub/datasets--snap-stanford--stark /root/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5 root@$IP:/workspace/.hf_home/hub/
ssh -p $PORT root@$IP 'cd /workspace/prime && ln -sfn p_joint_enc models/p_joint_head_enc; source /venv/main/bin/activate && uv pip install stark-qa sentence-transformers scipy pyahocorasick 2>&1 | tail -1'
echo "BOOTSTRAP_DONE $IP:$PORT"
