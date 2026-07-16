from pathlib import Path
import os

'''
加密ncnn模型的.bin\.param
注意: cpp加载时需要切换加载接口
'''

KEY = b"poly@2026_jdddp"



def xor_crypt(data: bytes) -> bytes:
    key_len = len(KEY)
    out = bytearray(data)

    for i in range(len(out)):
        out[i] ^= KEY[i % key_len] ^ ((i * 131 + 17) & 0xFF)

    return bytes(out)


def encrypt_file(src: str, dst: str):
    data = Path(src).read_bytes()
    Path(dst).write_bytes(xor_crypt(data))
    print(f"Done: {src} -> {dst}")

if __name__ == '__main__':
    # 你的专属密钥

    
    # 修改为你的实际文件名
    model_dir = r'D:\projects\20260525chongying\models\20260713_v2_best_ncnn_model'
    input_param = rf"{model_dir}\model.ncnn.param"
    output_param = rf"{model_dir}\model.ncnn.param.enc"
    
    input_bin = rf"{model_dir}\model.ncnn.bin"
    output_bin = rf"{model_dir}\model.ncnn.bin.enc"
    # print(input_param)
    if os.path.exists(input_param):
        encrypt_file(input_param, output_param)
    if os.path.exists(input_bin):
        encrypt_file(input_bin, output_bin)

'''
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace {
    const unsigned char kXorKey[] = "poly@2026_jdddp";

    std::vector<unsigned char> read_binary_file(const std::string& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            return {};
        }

        return std::vector<unsigned char>(
            std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>());
    }

    std::vector<unsigned char> xor_crypt(const std::vector<unsigned char>& data)
    {
        std::vector<unsigned char> out = data;
        const size_t key_len = sizeof(kXorKey) - 1;

        for (size_t i = 0; i < out.size(); ++i) {
            out[i] ^= kXorKey[i % key_len] ^
                static_cast<unsigned char>((i * 131 + 17) & 0xFF);
        }

        return out;
    }
}


// ---------------- load ----------------
bool YoloV8::load(const std::string& param,
    const std::string& bin,
    bool use_gpu)
{
    // 线程数和是否启用 Vulkan 在这里统一配置。
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = use_gpu;
    net.clear();

    std::cout << "[YoloV8::load] param path: " << param << std::endl;
    std::cout << "[YoloV8::load] bin path: " << bin << std::endl;
    std::cout << "[YoloV8::load] use_gpu: " << use_gpu << std::endl;

    const std::vector<unsigned char> encrypted_param = read_binary_file(param);
    const std::vector<unsigned char> encrypted_bin = read_binary_file(bin);
    if (encrypted_param.empty() || encrypted_bin.empty()) {
        std::cout << "[YoloV8::load] failed to read encrypted model files."
            << " encrypted_param.size=" << encrypted_param.size()
            << " encrypted_bin.size=" << encrypted_bin.size()
            << std::endl;
        return false;
    }

    std::cout << "[YoloV8::load] encrypted_param.size="
        << encrypted_param.size() << std::endl;
    std::cout << "[YoloV8::load] encrypted_bin.size="
        << encrypted_bin.size() << std::endl;

    const std::vector<unsigned char> decrypted_param_bytes = xor_crypt(encrypted_param);
    const std::vector<unsigned char> decrypted_bin_bytes = xor_crypt(encrypted_bin);

    std::cout << "[YoloV8::load] decrypted_param_bytes.size="
        << decrypted_param_bytes.size() << std::endl;
    std::cout << "[YoloV8::load] decrypted_bin_bytes.size="
        << decrypted_bin_bytes.size() << std::endl;

    // load_param_mem() 需要以 '\0' 结尾的文本参数串。
    decrypted_param_.assign(
        reinterpret_cast<const char*>(decrypted_param_bytes.data()),
        decrypted_param_bytes.size());
    decrypted_param_.push_back('\0');

    const size_t preview_len = std::min<size_t>(64, decrypted_param_bytes.size());
    std::string param_preview(
        reinterpret_cast<const char*>(decrypted_param_bytes.data()),
        preview_len);
    for (char& ch : param_preview) {
        if (ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
        }
    }
    std::cout << "[YoloV8::load] decrypted_param preview: "
        << param_preview << std::endl;
    if (decrypted_param_.rfind("7767517", 0) == 0) {
        std::cout << "[YoloV8::load] param magic looks valid." << std::endl;
    }
    else {
        std::cout << "[YoloV8::load] param magic mismatch, expected text param"
            << " starting with 7767517." << std::endl;
    }

    // load_model(const unsigned char*) 依赖外部内存，且要求至少 4 字节对齐。
    decrypted_model_storage_.assign((decrypted_bin_bytes.size() + 3) / 4, 0u);
    std::memcpy(decrypted_model_storage_.data(),
        decrypted_bin_bytes.data(),
        decrypted_bin_bytes.size());

    const int ret_param = net.load_param_mem(decrypted_param_.c_str());
    std::cout << "[YoloV8::load] load_param_mem ret=" << ret_param << std::endl;
    if (ret_param != 0) {
        return false;
    }

    const int ret_model = net.load_model(
        reinterpret_cast<const unsigned char*>(decrypted_model_storage_.data()));
    std::cout << "[YoloV8::load] load_model ret=" << ret_model << std::endl;
    if (ret_model <= 0) {
        return false;
    }

    std::cout << "[YoloV8::load] encrypted model load success." << std::endl;
    return true;
}
'''
