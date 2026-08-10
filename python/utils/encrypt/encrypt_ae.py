import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# 32字节密钥（AES-256）
KEY = bytes.fromhex(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)

def encrypt_file(src, dst):
    aes = AESGCM(KEY)

    # GCM 推荐 12 字节随机 nonce
    nonce = os.urandom(12)

    with open(src, "rb") as f:
        plaintext = f.read()

    ciphertext = aes.encrypt(nonce, plaintext, None)

    with open(dst, "wb") as f:
        # 保存格式：
        # [12字节 nonce][ciphertext + 16字节 tag]
        f.write(nonce)
        f.write(ciphertext)

    print(f"{src} -> {dst}")

encrypt_file("model.ncnn.param", "model.param.enc")
encrypt_file("model.ncnn.bin", "model.bin.enc")