#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void saveDetections(
    const std::string& image_name,
    const std::vector<Object>& objects,
    const std::string& save_path)
{
    json root;

    root["image"] = image_name;
    root["objects"] = json::array();

    for (const auto& obj : objects)
    {
        json item;

        item["label"] = obj.label;
        item["label_id"] = obj.label_id;
        item["score"] = obj.prob;

        item["bbox"] =
        {
            obj.rect.x,
            obj.rect.y,
            obj.rect.width,
            obj.rect.height
        };

        root["objects"].push_back(item);
    }

    std::ofstream ofs(save_path);

    ofs << root.dump(4);
}
/*
auto objects = detect(img);

saveDetections(
    "000001.jpg",
    objects,
    "results_fp32/000001.json");
*/