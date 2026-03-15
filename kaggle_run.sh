#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi
set -e

echo "==========================================="
echo "🚀 BƯỚC 1: Cập nhật Submodule và Cài đặt Thư viện"
echo "==========================================="
# Pull các submodule (nếu project bạn đang chứa submodule cho dataset/model)
echo "Đang cập nhật git submodules..."
git submodule update --init --recursive

echo "Đang cài đặt các thư viện từ requirements.txt..."
pip install -r requirements.txt

# (Tuỳ chọn) Nếu flow_matching cần cài từ GitHub, bỏ comment dòng dưới:
# pip install git+https://github.com/YOUR_USERNAME/flow_matching.git

echo ""
echo "==========================================="
echo "🔑 BƯỚC 2: Cấu hình Weights & Biases (W&B)"
echo "==========================================="
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️ CHÚ Ý: Biến môi trường WANDB_API_KEY chưa được thiết lập!"
    echo "Hệ thống sẽ chạy ở chế độ offline hoặc yêu cầu bạn nhập key thủ công nếu W&B bật."
    echo "Để tự động đăng nhập trên Kaggle, hãy chạy lệnh sau trước khi chạy script:"
    echo "export WANDB_API_KEY='key_của_bạn'"
else
    echo "Đang đăng nhập vào W&B..."
    wandb login $WANDB_API_KEY
fi

echo ""
echo "==========================================="
echo "🔥 BƯỚC 3: Bắt đầu quá trình Training"
echo "==========================================="
# Bạn có thể thay đổi các hyperparameter tại đây
python train.py \
    --epochs 10 \
    --batch_size 32 \
    --eval_batch_size 16 \
    --lr 5e-5

echo ""
echo "==========================================="
echo "✅ Hoàn tất quá trình Training!"
echo "Model và logs đã được lưu lại."
echo "==========================================="
