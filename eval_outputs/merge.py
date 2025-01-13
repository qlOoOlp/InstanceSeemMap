import pandas as pd
import json
files = ["/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/results1.json",
         "/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/results2.json",
         "/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/results3.json",
         "/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/results4.json"]

merged_data = {}

for idx in range(len(files)):
    file = files[idx]
    data = json.load(open(file))
    keys = list(data.keys())
    for key in keys:
        new_key = str(idx)+"_"+key
        vals = data[key]["results"][0]
        score = (vals["pacc"]+2*vals["mpacc"]+2*vals["miou"]+2*vals["fwmiou"])/7
        merged_data[new_key] = [data[key]["params"], vals, score, vals["num_embeddings"]]

return_data = sorted(merged_data, key=lambda x: x[2], reverse=True)
    
csv_data = []

for key in return_data:
    vals = merged_data[key][1]
    params = merged_data[key][0]
    entry = {"ID": key}
    entry.update({"score":merged_data[key][2], 
                  "num_instances":merged_data[key][3],
                  "pacc":vals["pacc"],
                  "mpacc":vals["mpacc"],
                  "miou":vals["miou"],
                  "fwmiou":vals["fwmiou"],
                  "top_k_auc":vals["top_k_auc"],
                  "top_k_auc_mpacc":vals["top_k_auc_mpacc"],
                  "min_depth":params["min_depth"],
                  "min_size_denoising_after_projection":params["min_size_denoising_after_projection"],
                  "threshold_pixelSize":params["threshold_pixelSize"],
                  "threshold_semSim":params["threshold_semSim"],
                  "threshold_bbox":params["threshold_bbox"],
                  "threshold_semSim_post":params["threshold_semSim_post"],
                  "threshold_geoSim_post":params["threshold_geoSim_post"],
                  "threshold_pixelSize_post":params["threshold_pixelSize_post"]
                  })
    csv_data.append(entry)
df = pd.DataFrame(csv_data).set_index("ID")
print(df)

return_file = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/total_result.txt"
return_file2 = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/total_result.xlsx"
with open(return_file, "w") as f:
    for key in return_data:
        f.write(key + "\n")
        f.write(str(merged_data[key][0]) + "\n")
        f.write(str(merged_data[key][1]) + "\n")
        f.write(str(merged_data[key][2]) + "\n")
        f.write(str(merged_data[key][3]) + "\n")
        f.write("\n")


df.to_excel(return_file2)#, index=False)