# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.lightning.pytorch.plugins import MegatronDataSampler


class MockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_length: int = 2048,
        decoder_seq_length: Optional = None,
        vocab_size: int = 128256,
        crop_size: Tuple[int, int] = (560, 560),
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.vocab_size = vocab_size
        self.crop_size = crop_size

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = _MockMLlamaDataset(
            self.vocab_size, self.crop_size, "train", self.num_train_samples, self.decoder_seq_length
        )
        self._validation_ds = _MockMLlamaDataset(
            self.vocab_size, self.crop_size, "valid", self.num_val_samples, self.decoder_seq_length
        )
        self._test_ds = _MockMLlamaDataset(
            self.vocab_size, self.crop_size, "test", self.num_test_samples, self.decoder_seq_length
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


class _MockMLlamaDataset(Dataset):
    def __init__(
        self,
        vocab_size,
        crop_size,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length

        self.vocab_size = vocab_size

        self.image_height, self.image_width = crop_size

        self.length = num_samples
        self.seed = seed

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Generate data of the expected size and datatype (based on GPTDataset).
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length + 1], dtype=np.int64))
        images = torch.from_numpy(np_gen.standard_normal((1, 4, 3, self.image_height, self.image_width)))
        aspect_ratio_ids = torch.from_numpy(np_gen.integers(8, size=[1], dtype=np.int64)) + 1

        labels = tokens.clone()
        tokens = tokens[:-1]
        labels = labels[1:]

        return {
            "images": images,
            "masks": [[5, 512]],
            "num_chunks": [4],
            "tokens": tokens,
            "aspect_ratio_ids": aspect_ratio_ids,
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
            "labels": labels,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        collated_batch = {}
        collated_batch["batch_masks"] = [sample.pop("masks") for sample in batch]
        collated_batch["attention_mask"] = None
        collated_batch.update(data.dataloader.default_collate(batch))
        collated_batch["batch_images"] = collated_batch.pop("images")
        return collated_batch

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)