import datetime
import os
import sys
sys.path.append("../implementations")
from scripts.tools import utils
from scripts.generation.deduplicate_model import ModelDeduplication
import time
from scripts.coverage.architecture_coverage import ArchitectureMeasure
import dill


os.environ["CUDA_VISIBLE_DEVICES"]="2"
origin_model_path = "../data/origin_models"
output_path = "../data/synthesis_analysis"
if not os.path.exists(output_path):
    os.makedirs(output_path)
count = 0
TOTAL_COUNT = 20
analysis_result = []
while count < TOTAL_COUNT:
    start_time = datetime.datetime.now()
    experiment_path = os.path.join(output_path, f"{count}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    print(f"\n\n ============================= Iteration: {count} ===============================\n\n")
    origin_analyze_result = {}
    new_analyze_result = {}
    architecture_measure = ArchitectureMeasure()
    for path in os.listdir(origin_model_path):
        if not path.endswith(".h5"):
            continue
        model_name = path.split(".h5")[0]
        import keras
        origin_model = keras.models.load_model(os.path.join(origin_model_path, path))
        origin_analyze_result[model_name] = {}
        new_analyze_result[model_name] = {}
        print(f"=============== Working On Model: {path} ===================")
        s_time = time.time()
        new_model, origin_diversity, new_diversity = None, None, None
        e_time = time.time()
        while e_time - s_time <= 60*10 and new_model is None:
            try:
                modelDeduplication = ModelDeduplication()
                new_model, origin_diversity, new_diversity = modelDeduplication.run(os.path.join(origin_model_path, path))
            except:
                # print(traceback.format_exc())
                new_model = None
            e_time = time.time()
        if new_model is None:
            print(f"Failed When Synthesizing Model: {model_name}")
        architecture_measure.update_diversity(new_model, "fake_name")
        new_analyze_result[model_name]["num_layers"] = len(new_model.layers)
        new_analyze_result[model_name]["num_weights"] = new_model.count_params()
        new_analyze_result[model_name]["synthesis_time"] = e_time - s_time

        model_json = new_model.to_json()
        with open(os.path.join(experiment_path, f"{path[:-5]}-simplified.json"), "w") as file:
            file.write(model_json)

    api_cov, api_pair_cov, config_cov, input_cov, ndims_cov, dtype_cov, shape_cov = architecture_measure.coverage()
    new_analyze_result["coverage"] = {}
    new_analyze_result["coverage"]["layer_inputs_cov"] = input_cov
    new_analyze_result["coverage"]["layer_params_cov"] = config_cov
    new_analyze_result["coverage"]["layer_sequences_cov"] = api_pair_cov
    analysis_result.append(new_analyze_result)
    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h, m, s = utils.ToolUtils.get_HH_mm_ss(time_delta)
    print(f"Model Deduplication Is Finished: Time used: {h} hour,{m} min,{s} sec")
    count += 1

    analysis_result_path = os.path.join(output_path, "analysis_result.pkl")
    with open(analysis_result_path, "wb") as file:
        dill.dump(analysis_result, file)
