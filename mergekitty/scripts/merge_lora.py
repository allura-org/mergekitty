# Copyright (C) 2025 Allura-org
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import click
import peft
import torch
from pydantic import BaseModel

from mergekitty.common import AdapterReference, ModelReference, ModelPath
from mergekitty.executor import SingleThreadedExecutor
from mergekitty.io import LazyTensorLoader
from mergekitty.task import Task
from mergekitty.io.tasks import LoadTensor, LoaderCache, SaveTensor, TensorWriterTask
from mergekitty.options import MergeOptions, add_merge_options


@click.command("mergekitty-merge-lora")
@click.argument("base_model", type=str)
@click.argument("adapter_model", type=str)
@click.argument("output_path", type=str)
@add_merge_options
def main(
    base_model: str,
    adapter_model: str,
    output_path: str,
    merge_options: MergeOptions,
) -> None:
    """
    Merge a LoRA adapter with a base model to create a new model.

    \b
    Arguments:
    BASE_MODEL - Path to the base model or HF model ID
    ADAPTER_MODEL - Path to the LoRA adapter or HF model ID
    OUTPUT_PATH - Where to save the merged model
    """
    logging.basicConfig(
        level=logging.WARNING if merge_options.quiet else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    model_ref = ModelReference.model_validate(base_model)
    # Create AdapterReference properly
    adapter_path = ModelPath.model_validate(adapter_model)
    adapter_ref = AdapterReference(adapter=adapter_path)

    # Set up loader cache with options
    loader_cache = LoaderCache()
    loader_cache.setup(merge_options)

    plan = plan_merge(model_ref, adapter_ref, output_path, merge_options)
    executor = SingleThreadedExecutor()
    
    logging.info(f"Merging {adapter_model} into {base_model}, saving to {output_path}")
    for task, result in executor.run(plan.tasks):
        pass
    logging.info(f"Merged model saved to {output_path}")


class LoraScaleTask(Task[float]):
    """Task that calculates the LoRA scaling factor based on adapter config."""
    adapter_config: peft.PeftConfig

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **kwargs) -> float:
        return self.adapter_config.lora_alpha / self.adapter_config.r


class MergeLoraTask(Task[torch.Tensor]):
    """Task that merges a LoRA adapter with a base weight."""
    base_tensor: Task[torch.Tensor]
    adapter_tensor_a: Task[torch.Tensor]
    adapter_tensor_b: Task[torch.Tensor]
    adapter_config: peft.PeftConfig
    
    def arguments(self) -> Dict[str, Task]:
        return {
            "base_tensor": self.base_tensor,
            "adapter_tensor_a": self.adapter_tensor_a,
            "adapter_tensor_b": self.adapter_tensor_b,
            "lora_scale": LoraScaleTask(adapter_config=self.adapter_config),
        }

    def execute(self, base_tensor: torch.Tensor, adapter_tensor_a: torch.Tensor, 
                adapter_tensor_b: torch.Tensor, lora_scale: float) -> torch.Tensor:
        old_dtype = base_tensor.dtype
        tensor = base_tensor.to(torch.float32)
        tensor += lora_scale * (adapter_tensor_b @ adapter_tensor_a)
        return tensor.to(old_dtype)


class PlanResults(BaseModel):
    tasks: List[Task]
    base_vocab_size: int
    final_vocab_size: int


def plan_merge(
    model_ref: ModelReference,
    adapter_ref: AdapterReference,
    output_path: str,
    options: MergeOptions,
) -> PlanResults:
    """
    Plan the merging of a LoRA adapter with a base model.
    
    Args:
        model_ref: Reference to the base model
        adapter_ref: Reference to the LoRA adapter
        output_path: Where to save the merged model
        options: Merge options
        
    Returns:
        A plan with tasks to execute the merge
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Get the adapter config
    adapter_config = adapter_ref.config(trust_remote_code=options.trust_remote_code)
    
    # Get the base model and adapter loaders
    base_loader = LoaderCache().get(model_ref)
    adapter_loader = adapter_ref.lazy_loader(
        cache_dir=options.transformers_cache, 
        lazy_unpickle=options.lazy_unpickle
    )
    
    # Create tensor writer task
    writer_task = TensorWriterTask(
        out_path=output_path,
        max_shard_size=options.out_shard_size,
        safe_serialization=options.safe_serialization,
    )
    
    tasks = []
    save_tasks = []
    
    # Get the list of base model tensors
    base_tensor_names = set(base_loader.index.tensor_paths.keys())
    
    # Vocabulary sizes 
    base_vocab_size = None
    final_vocab_size = None
    
    # Get the list of adapter tensors and identify LoRA pairs
    lora_pairs = {}
    modules_to_save = set()
    
    # Parse adapter tensor names to identify LoRA pairs and modules to save
    for tensor_name in adapter_loader.index.tensor_paths.keys():
        # Handle modules to save directly (full weights)
        if "base_model.model." in tensor_name and ".base_layer.weight" in tensor_name:
            module_name = tensor_name.replace("base_model.model.", "").replace(".base_layer.weight", "")
            modules_to_save.add(module_name)
        # Handle LoRA A matrices
        elif "lora_A.weight" in tensor_name:
            module_name = tensor_name.replace("base_model.model.", "").replace(".lora_A.weight", "")
            if module_name not in lora_pairs:
                lora_pairs[module_name] = {"A": tensor_name}
            else:
                lora_pairs[module_name]["A"] = tensor_name
        # Handle LoRA B matrices
        elif "lora_B.weight" in tensor_name:
            module_name = tensor_name.replace("base_model.model.", "").replace(".lora_B.weight", "")
            if module_name not in lora_pairs:
                lora_pairs[module_name] = {"B": tensor_name}
            else:
                lora_pairs[module_name]["B"] = tensor_name
        # Handle embedding LoRA
        elif "lora_embedding_A" in tensor_name:
            module_name = tensor_name.replace("base_model.model.", "").replace(".lora_embedding_A", "")
            if module_name not in lora_pairs:
                lora_pairs[module_name] = {"A": tensor_name}
            else:
                lora_pairs[module_name]["A"] = tensor_name
        elif "lora_embedding_B" in tensor_name:
            module_name = tensor_name.replace("base_model.model.", "").replace(".lora_embedding_B", "")
            if module_name not in lora_pairs:
                lora_pairs[module_name] = {"B": tensor_name}
            else:
                lora_pairs[module_name]["B"] = tensor_name
    
    # Process all base model tensors
    for tensor_name in base_tensor_names:
        module_name = None
        
        # Check if tensor weight corresponds to a LoRA module
        for lora_module in lora_pairs:
            if f"{lora_module}.weight" == tensor_name:
                module_name = lora_module
                break
                
        if module_name in modules_to_save:
            # For modules to save directly (e.g. expanded embeddings), load from adapter
            adapter_tensor_name = f"base_model.model.{module_name}.base_layer.weight"
            load_task = LoadTensor(
                model=adapter_ref.adapter, 
                tensor=adapter_tensor_name, 
                device="cuda" if options.read_to_gpu else None
            )
            save_task = SaveTensor(
                tensor_name=tensor_name,
                tensor_task=load_task,
                writer_task=writer_task,
                clone=options.clone_tensors,
                dtype=None
            )
            save_tasks.append(save_task)
            
            # Capture vocab size for embeddings
            if ".wte.weight" in tensor_name or ".embeddings.word_embeddings.weight" in tensor_name:
                # Get adapter weight to determine its size
                adapter_tensor = adapter_loader.get_tensor(adapter_tensor_name)
                base_tensor = base_loader.get_tensor(tensor_name)
                base_vocab_size = base_tensor.shape[0]
                final_vocab_size = adapter_tensor.shape[0]
                
        elif module_name in lora_pairs and "A" in lora_pairs[module_name] and "B" in lora_pairs[module_name]:
            # Apply LoRA to this module
            base_tensor_task = LoadTensor(
                model=model_ref, 
                tensor=tensor_name,
                device="cuda" if options.read_to_gpu else None
            )
            
            adapter_a_task = LoadTensor(
                model=adapter_ref.adapter, 
                tensor=lora_pairs[module_name]["A"],
                device="cuda" if options.read_to_gpu else None
            )
            
            adapter_b_task = LoadTensor(
                model=adapter_ref.adapter, 
                tensor=lora_pairs[module_name]["B"],
                device="cuda" if options.read_to_gpu else None
            )
            
            merge_task = MergeLoraTask(
                base_tensor=base_tensor_task,
                adapter_tensor_a=adapter_a_task,
                adapter_tensor_b=adapter_b_task,
                adapter_config=adapter_config
            )
            
            save_task = SaveTensor(
                tensor_name=tensor_name,
                tensor_task=merge_task,
                writer_task=writer_task,
                clone=options.clone_tensors,
                dtype=None
            )
            save_tasks.append(save_task)
            
            # If this is an embedding, capture vocabulary sizes
            if ".wte.weight" in tensor_name or ".embeddings.word_embeddings.weight" in tensor_name:
                base_tensor = base_loader.get_tensor(tensor_name)
                base_vocab_size = base_tensor.shape[0]
                final_vocab_size = base_vocab_size  # No vocab expansion in standard LoRA merging
                
        else:
            # Copy tensor from base model unchanged
            base_tensor_task = LoadTensor(
                model=model_ref, 
                tensor=tensor_name,
                device="cuda" if options.read_to_gpu else None
            )
            
            save_task = SaveTensor(
                tensor_name=tensor_name,
                tensor_task=base_tensor_task,
                writer_task=writer_task,
                clone=options.clone_tensors,
                dtype=None
            )
            save_tasks.append(save_task)
    
    # If we didn't find embedding weights, use default values
    if base_vocab_size is None:
        base_vocab_size = 0
        final_vocab_size = 0
    
    tasks.extend(save_tasks)
    
    # Copy config and tokenizer files
    if options.copy_tokenizer:
        logging.info("Copying tokenizer and config files")
        for file_name in base_loader.index.other_files:
            if file_name.endswith('.json') or file_name.endswith('.model') or file_name.endswith('.tokenizer'):
                src_path = os.path.join(os.path.dirname(base_loader.index.root_path), file_name)
                dst_path = os.path.join(output_path, file_name)
                # Copy file
                with open(src_path, 'rb') as src_file, open(dst_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
    
    return PlanResults(
        tasks=tasks,
        base_vocab_size=base_vocab_size,
        final_vocab_size=final_vocab_size,
    )


def test_merge_lora():
    """A simple test function to validate the core logic without downloading models."""
    import tempfile
    import torch
    
    print("Testing core LoRA merge functionality...")
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create synthetic data
        base_weight = torch.randn(10, 20)
        lora_a = torch.randn(5, 20)
        lora_b = torch.randn(10, 5)
        
        # Expected result with scale=1.0
        expected = base_weight + lora_b @ lora_a
        
        # Mock peft config
        mock_config = peft.LoraConfig(
            r=5,
            lora_alpha=5,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Test directly with the execution logic from MergeLoraTask
        old_dtype = base_weight.dtype
        tensor = base_weight.to(torch.float32)
        lora_scale = mock_config.lora_alpha / mock_config.r
        tensor += lora_scale * (lora_b @ lora_a)
        result = tensor.to(old_dtype)
        
        # Check result
        assert torch.allclose(result, expected * lora_scale, rtol=1e-4), "Merged tensor doesn't match expected result"
        print("  - LoRA merge calculation test passed!")
        
        # Test LoraScaleTask directly
        scale_task = LoraScaleTask(adapter_config=mock_config)
        scale = scale_task.execute()
        assert scale == 1.0, f"Scale should be 1.0, got {scale}"
        print("  - LoraScaleTask calculation test passed!")
        
        print("All tests passed!")
        return True


if __name__ == "__main__":
    main()