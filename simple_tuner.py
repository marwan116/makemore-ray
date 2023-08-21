
import torch
import time
import torch.nn as nn
import argparse
import ray
from ray import train, tune
from ray.tune.tuner import Tuner, TuneConfig
from ray.air import session, Checkpoint
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(layer_size, output_size)

    def forward(self, input):
        return self.layer2(self.relu(self.layer1(input)))


def train_loop_per_worker(config):
    dataset_shard = session.get_dataset_shard("train")
    model = NeuralNetwork(
        input_size=config["input_size"],
        layer_size=config["layer_size"],
        output_size=config["output_size"],
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    model = train.torch.prepare_model(model)

    for epoch in range(config["num_epochs"]):
        for batches in dataset_shard.iter_torch_batches(
            batch_size=32, dtypes=torch.float
        ):
            inputs, labels = torch.unsqueeze(batches["x"], 1), batches["y"]
            output = model(inputs)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        session.report(
            {"loss": loss.item()},
            # note checkpointing requires s3 storage path
            # checkpoint=Checkpoint.from_dict(
            #     dict(epoch=epoch, model=model.state_dict())
            # ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address")
    args = parser.parse_args()
    ray.init(address=args.address)

    use_gpu = False
    input_size = 1
    layer_size = 15
    output_size = 1
    num_epochs = 3
    lr = 1e-3

    train_dataset = ray.data.from_items([{"x": x, "y": 2 * x + 1} for x in range(200)])
    scaling_config = ScalingConfig(num_workers=3, use_gpu=use_gpu)
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "input_size": input_size,
            "layer_size": layer_size,
            "output_size": output_size,
            "num_epochs": num_epochs,
            "lr": lr,
        },
        scaling_config=scaling_config,
        datasets={"train": train_dataset},
    )

    param_space = {"train_loop_config": {"lr": tune.loguniform(0.0001, 0.01)}}

    tuner = Tuner(
        trainer,
        param_space=param_space,
        tune_config=TuneConfig(num_samples=5, metric="loss", mode="min"),
    )

    # Execute tuning.
    result_grid = tuner.fit()

    # Fetch the best result.
    best_result = result_grid.get_best_result()
    print("Best Result:", best_result)
