import argparse
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pl_datamodule_trf import TokenClassificationDataModule
from pl_module_trf import TokenClassificationModule


def make_common_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--data_dir",
        default="/app/workspace/data",
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        help="Path to pretrained model config (for transformers)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        type=str,
        help="Path to pretrained tokenzier JSON config (for transformers)",
    )
    parser.add_argument(
        "--labels",
        default="workspace/data/label_types.txt",
        type=str,
        help="Path to a file containing all labels. (for transformers)",
    )
    parser.add_argument("--nbest", default=1, type=int, help="CNN(to be implemented)")
    parser.add_argument("--number_normalized", default=True, type=bool, help="CNN")
    return parser


def build_args(notebook=False):
    parser = make_common_args()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = TokenClassificationModule.add_model_specific_args(parent_parser=parser)
    parser = TokenClassificationDataModule.add_model_specific_args(parent_parser=parser)
    if not notebook:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=[])
    args.delimiter = " "
    args.is_bio = False
    return args


def save_pickle_in_module(
    save_path: str,
    model_path: str,
    vocab_path: str,
    config_path: str,
    notebook: bool = False,
):
    """Save pickle in the current module hierarchy from pytorch model.
    NOTE: アプリケーションコンテナ内でpickleを実行しないとpickle対象のクラスパスが解決できない。
    """
    args = build_args(notebook)
    args.model_path = model_path
    args.vocab_path = vocab_path
    args.config_path = config_path
    args.do_train = False
    args.do_predict = True
    args.gpu = False
    pl_module = TokenClassificationModule(args)
    pl_module.save_pickle(save_path)


def main_as_plmodule():
    """PyTorch-Lightning Moduleとして訓練・予測を行う"""

    args = build_args()
    args.gpu = torch.cuda.is_available()

    pl.seed_everything(args.seed)
    Path(args.output_dir).mkdir(exist_ok=True)
    print(f"Building dataset...")
    dm = TokenClassificationDataModule(args)
    dm.prepare_data()

    if args.do_train:
        print(f"Start training...")
        dm.setup(stage="fit")
        # make pl module
        model = TokenClassificationModule(args, bilou=dm.bilou, use_datasets=dm.use_datasets)
        # make trainer with callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename="checkpoint-{epoch}-{val_loss:.2f}"
            if args.monitor == "loss"
            else "checkpoint-{epoch}-{val_f1:.2f}",
            monitor="val_loss" if args.monitor == "loss" else "val_f1",
            mode="min" if args.monitor == "loss" else "max",
            save_top_k=10,
            verbose=True,
        )
        lr_logger = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[lr_logger, checkpoint_callback],
            deterministic=True,
            accumulate_grad_batches=args.accumulate_grad_batches,
        )

        # train
        trainer.fit(model, dm)
        # test
        trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_predict:
        if args.model_path.endswith(".ckpt"):
            ## NOTE: the path structure on training is pickled in .ckpt
            print(args.model_path)
            model = TokenClassificationModule.load_from_checkpoint(args.model_path)
            save_path = Path(args.model_path).parent / "prediction_model.pt"
            torch.save(model.model.state_dict(), save_path)

            trainer = pl.Trainer.from_argparse_args(args, deterministic=True)
            trainer.test(
                model=model,
                ckpt_path=args.model_path,
                test_dataloaders=dm.test_dataloader(),
            )

            args.model_path = str(save_path)
        else:
            model = TokenClassificationModule(args)

        save_path = Path(args.model_path).parent / "prediction_model.pkl"
        model.save_pickle(save_path)
        datadir = Path(args.data_dir)
        if datadir.exists():
            if (datadir / "test.txt").exists():
                dl = dm.test_dataloader()
                fexamples = dm.test_dataset.fexamples
            else:
                texts = []
                for txt_path in datadir.glob("*.txt"):
                    with open(txt_path) as fp:
                        text = fp.read()
                        texts.append(text)
                dl = dm.get_prediction_dataloader(texts)
                fexamples = dm.dataset.fexamples

            print(f"Start prediction...")
            time_start = time.time()

            prediction_batch = [
                model.predict_step(batch, i) for i, batch in enumerate(dl)
            ]
            content_list = [w for d in prediction_batch for w in d["input"]]
            decode_results = [l for d in prediction_batch for l in d["prediction"]]
            # print(len(content_list), len(decode_results))
            # print(len(dm.dataset), len(dm.dataset.fexamples))

            time_finish = time.time()
            timecost = time_finish - time_start
            print(f"End: {timecost} sec.")
            # TODO: do alignment with original tokens
            outpath = Path(args.output_dir) / "result.txt"
            print(content_list[0])
            print(decode_results[0])
            # model.write_decoded_results(content_list, decode_results, outpath)
            sent_num = len(decode_results)
            assert sent_num == len(content_list)
            with open(outpath, "w") as fout:
                for idx in range(sent_num):
                    sent_length = len(decode_results[idx])
                    for idy in range(sent_length):
                        fout.write(
                            "{} {} {}\n".format(
                                fexamples[idx].words[idy],
                                content_list[idx][idy],
                                decode_results[idx][idy],
                            )
                        )
                    fout.write("\n")
        else:
            print(f"no input file given: {datadir}")
            exit(0)


if __name__ == "__main__":
    main_as_plmodule()
