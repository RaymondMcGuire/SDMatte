cd ..

uv run python infer_single.py --config-dir configs/SDMatte.py --checkpoint-dir "F:/huggingface/SDMatte/SDMatte.pth" --image customdata/images/0001.png --output output/res.png --auto-bbox

pause