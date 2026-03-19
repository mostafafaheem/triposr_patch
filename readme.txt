1. modified from https://github.com/VAST-AI-Research/TripoSR
2. wget https://huggingface.co/stabilityai/TripoSR/blob/main/model.ckpt ./ckpt/TripoSR/model.ckpt
3. put your .glb files under /data/furniture/[Source]/[Category]/, and run .py under ./code/: render.py -> focus.py -> split.py
   - These scripts  support nested directory structures automatically.
4. check code in train_infer.py, comment or uncomment 'infer/train' lines.
5. install dependencies with `pip install -r requirements.txt`. (see network.py and code/focus.py for additional installation hints, 
    or just use modal image if you're into that).

#logic correct, handles multiple sources and categories, supports full checkpoint training.

have fun.
