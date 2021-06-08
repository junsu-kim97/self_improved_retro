import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split big dataset and save")
    parser.add_argument("--input_path", default='./retro_star/dataset/routes_train.pkl',
                        type=str)
    parser.add_argument("--output_path", default='./retro_star/dataset/train_routes_shards/',
                        type=str)
    parser.add_argument("--unit_size", default=25000,
                        type=int)
    args = parser.parse_args()

    source_dataset = pickle.load(open(args.input_path, 'rb'))
    size = args.unit_size
    assert len(source_dataset) >= size
    cnt = 0
    for i in range(0, len(source_dataset), size):
        shard = source_dataset[i:i + size]

        save_path = args.output_path + "shard_" + str(cnt) + ".pkl"
        with open(save_path, "wb") as f:
            pickle.dump(shard, f)

        print("saved @ ", save_path)
        cnt += 1

    print("done")
