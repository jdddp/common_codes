#!/usr/bin/env python3
import hashlib
import os
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# 直接在這裡改你的模型路徑。
INPUT_PARAM = r"./model.ncnn.param"
OUTPUT_PARAM = r"./model.ncnn.param.enc"

INPUT_BIN = r"./model.ncnn.bin"
OUTPUT_BIN = r"./model.ncnn.bin.enc"

MAGIC = b"QTAESv01"
IV_LEN = 16


def derive_key() -> bytes:
    # 這套派生邏輯要和 Qt/QAESEncryption 端保持完全一致。
    passphrase = b"poly::2026::jdddp::ncnn::qtaes::cbc::pkcs7"
    salt = b"poly_qtaes@2026#jdddp$ncnn%secure"

    round1 = hashlib.sha256(passphrase).digest()
    round2 = hashlib.sha256(salt + round1 + passphrase[::-1]).digest()
    round3 = hashlib.sha256(round1 + b"::" + salt + b"::" + round2 + b"::model_protect").digest()
    return round3


def encrypt_bytes(data: bytes) -> bytes:
    key = derive_key()
    iv = os.urandom(IV_LEN)

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(data) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return MAGIC + iv + ciphertext


def encrypt_file(src: Path, dst: Path) -> None:
    plaintext = src.read_bytes()
    encrypted = encrypt_bytes(plaintext)
    dst.write_bytes(encrypted)
    print(f"encrypted: {src} -> {dst}")


def main() -> None:
    encrypt_file(Path(INPUT_PARAM), Path(OUTPUT_PARAM))
    encrypt_file(Path(INPUT_BIN), Path(OUTPUT_BIN))


if __name__ == "__main__":
    main()
