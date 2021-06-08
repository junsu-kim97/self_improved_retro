import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge gold.csv, templates.dat")
    parser.add_argument("--gold_csv_data", "-g", nargs='+')
    parser.add_argument("--tpl_data", "-t", nargs='+')

    parser.add_argument("--save_path_gold_csv", default="", type=str)
    parser.add_argument("--save_path_tpl", default="", type=str)

    args = parser.parse_args()

    gold_csv_list = args.gold_csv_data
    tpl_data_list = args.tpl_data

    merged_gold_csv = None
    merged_tpl_data = []
    for gold_csv in gold_csv_list:
        df = pd.read_csv(gold_csv)
        if merged_gold_csv is None:
            merged_gold_csv = df
        else:
            merged_gold_csv = pd.concat([merged_gold_csv, df], axis=0)

    for tpl_data in tpl_data_list:
        with open(tpl_data, "r") as f:
            while True:
                line = f.readline().rstrip()
                if not line:
                    break
                merged_tpl_data.append(line)

    merged_gold_csv.to_csv(args.save_path_gold_csv, index=False)

    with open(args.save_path_tpl, "w") as f:
        f.write("\n".join(merged_tpl_data))
    print("done")




